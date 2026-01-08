export PYTHONPATH=$(pwd)


# CUDA_VISIBLE_DEVICES=0,1,3,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=7777 train.py \
#   --config_path configs/dmd_wan22.yaml \
#   --save_dir outputs/dmd_wan22_flow \


# CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --master_port=9999  train.py \
#   --config_path configs/dmd_wan22.yaml \
#   --save_dir outputs/dmd_wan22_flow \


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=7777 train.py \
  --config_path configs/self_forcing_dmd_wan22_resume2.yaml \
  --save_dir outputs/self_forcing_dmd_wan22_resume_flow2 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=7777 train.py \
#   --config_path configs/self_forcing_dmd_wan21_resume.yaml \
#   --save_dir outputs/self_forcing_dmd_wan21_resume \

# CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --master_port=9999  train.py \
#   --config_path configs/self_forcing_dmd.yaml \
#   --save_dir outputs/self_forcing_dmd \
