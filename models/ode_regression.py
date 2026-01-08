import torch.nn.functional as F
from typing import Tuple
import torch
from torch import nn

from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from backbones.util import masks_like
import torch.distributed as dist


class ODERegression(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__()
        self.args = args

        self.model_name = self.args.model_name
        self.is_causal = args.generator_type == "causal"
        self.dtype = torch.bfloat16 if self.args.mixed_precision else torch.float32
        self.device = device

        self._initialize_models()

        # initialize for causal
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

    def _initialize_models(self):
        self.generator = WanDiffusionWrapper(**getattr(self.args, "model_kwargs", {}),
                                            model_name=self.model_name,
                                            is_causal=self.is_causal,
                                            local_attn_size=self.args.local_attn_size,
                                        )
        self.generator.model.requires_grad_(True)
        if self.args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.text_encoder = WanTextEncoder(model_name=self.model_name)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=self.model_name)
        self.vae.requires_grad_(False)

        # initialize schedule
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        if hasattr(self.args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(self.args.denoising_step_list, dtype=torch.long)
            if self.args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list].to(self.device)

    def _process_timestep(self, timestep):
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.

        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        # make the noise level the same within every block
        timestep = timestep.reshape(
            timestep.shape[0], -1, self.num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep


    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = torch.randint(0, len(self.denoising_step_list), [
            batch_size, num_frames], device=self.device, dtype=torch.long)

        index = self._process_timestep(index)

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width)
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        # breakpoint()
        if "2.2" in self.generator.model_name:
            # 这里如果不用图片，zero应设置为false
            mask1, mask2 = masks_like(noisy_input, zero=False if conditional_dict.get("wan22_image_latent", None) is None else True)
            mask2 = torch.stack(mask2, dim=0)
            if conditional_dict.get("wan22_image_latent", None) is not None:
                noisy_input = (1. - mask2) * conditional_dict["wan22_image_latent"] + mask2 * noisy_input
                noisy_input = noisy_input.to(self.device, dtype=self.dtype)

            temp_ts = (mask2[:, :, 0, ::2, ::2] * timestep.unsqueeze(-1).unsqueeze(-1).to(device=self.device, dtype=self.dtype))
            temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
            # temp_ts = torch.cat([temp_ts, temp_ts.new_ones(self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep], dim=1)
            wan22_input_timestep = temp_ts.to(self.device, dtype=torch.long)
        else:
            mask1, mask2 = None, None
            wan22_input_timestep = None
        # append anything
        conditional_dict['mask2'] = mask2
        conditional_dict['wan22_input_timestep'] = wan22_input_timestep

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        return loss, log_dict
