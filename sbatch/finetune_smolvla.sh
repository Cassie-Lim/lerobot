#!/bin/bash
#SBATCH --job-name=finetune_smolvla
#SBATCH --output=logs/finetune_smolvla_lmexpert_%j.out
#SBATCH --error=logs/finetune_smolvla_lmexpert_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop"

# Load conda - try miniforge3 first, fallback to miniconda3
if [ -f /coc/flash7/mlin365/miniforge3/etc/profile.d/conda.sh ]; then
    source /coc/flash7/mlin365/miniforge3/etc/profile.d/conda.sh
elif [ -f /coc/flash7/mlin365/miniconda3/etc/profile.d/conda.sh ]; then
    source /coc/flash7/mlin365/miniconda3/etc/profile.d/conda.sh
fi

# Activate environment
conda activate lerobot


# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python src/lerobot/scripts/lerobot_train_val.py \
#     --policy.path=lerobot/smolvla_base \
#     --policy.repo_id=cassielin0910/my_libero_smolvla \
#     --dataset.repo_id=lerobot/libero_spatial_image \
#     --policy.output_features=null \
#     --policy.input_features=null \
#     --policy.optimizer_lr=1e-3 \
#     --policy.scheduler_decay_lr=1e-4 \
#     --env.type=libero \
#     --env.task=libero_spatial \
#     --eval.batch_size=1 \
#     --eval.n_episodes=1 \
#     --eval_freq=1000 \
#     --log_freq=10 \
#     --steps=100_000 \
#     --batch_size=32 \
#     --peft.method_type=LORA \
#     --peft.r=16 \
#     --wandb.enable=true \
#     --wandb.project=smolvla_libero \
#     --wandb.entity=cassielin0910


python src/lerobot/scripts/lerobot_train_val.py \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=cassielin0910/my_libero_smolvla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --policy.output_features=null \
    --policy.input_features=null \
    --policy.optimizer_lr=1e-3 \
    --policy.scheduler_decay_lr=1e-4 \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --eval_freq=1000 \
    --log_freq=10 \
    --steps=100_000 \
    --batch_size=32 \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.target_modules='(model\.vlm_with_expert\.lm_expert\..*\.(down|gate|up)_proj|.*\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))' \
    --wandb.enable=true \
    --wandb.project=smolvla_libero \
    --wandb.entity=cassielin0910


