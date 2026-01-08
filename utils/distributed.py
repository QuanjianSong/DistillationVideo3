from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from accelerate import Accelerator
from typing import Dict
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    StateDictOptions,
)

def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint

# def fsdp_state_dict(model, opts=None):
#     opts = StateDictOptions(
#         full_state_dict=True,
#         cpu_offload=True,
#     )

#     model_state, _ = get_state_dict(model, None, options=opts)
#     return model_state


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def configure_fsdp_plugin(config):
    # 1) mixed precision 对齐
    mixed_precision_policy = None
    if getattr(config, "mixed_precision", False):
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    # breakpoint()
    # 2) auto wrap policy 对齐（并保持和第一段一样的报错语义）
    wrap_strategy = getattr(config, "generator_fsdp_wrap_strategy", "size")
    if wrap_strategy == "transformer":
        transformer_module = getattr(config, "transformer_module", None)
        if transformer_module is None:
            raise ValueError("Transformer wrap strategy requires 'transformer_module' in config.")
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        min_num_params = getattr(config, "min_num_params", int(5e7))
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    # 3) NCCL 环境变量副作用对齐
    os.environ["NCCL_CROSS_NIC"] = "1"

    # 4) sharding strategy 对齐
    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[getattr(config, "sharding_strategy", "full")]

    # 5) cpu offload 对齐
    cpu_offload = CPUOffload(offload_params=getattr(config, "cpu_offload", False))

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        # mixed_precision_policy=mixed_precision_policy,
        cpu_offload=cpu_offload,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=False,
    )
    return fsdp_plugin


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    torch.cuda.set_device(local_rank)

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
                        rank=rank,
                        world_size=world_size,
                        backend=backend,
                        init_method=init_method,
                        timeout=timedelta(minutes=30),
                        device_id=local_rank,)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.named_parameters():
                self.shadow[n].mul_(self.decay).add_(p.detach().float().cpu(), alpha=1. - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
