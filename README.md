# Installation

```python
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge
conda install cmake==3.5.0
git clone git@github.com:Cassie-Lim/lerobot.git
cd lerobot
pip install -e ".[smolvla,peft,libero]"
export MUJOCO_GL=egl

```
# Libero Aata Analysis

Use jupyter notebook `examples/dataset/libero_dataset_inspection.ipynb` to download libero datasets and check the format of data.

# Data Preprocessing

Run `src/lerobot/scripts/lerobot_prepare_train_val_splits.py` to generate the splits used in eval.

# Train and eval

Use following command for training:

```python

python src/lerobot/scripts/lerobot_train_val.py \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=[HUGGINGFACE_USER_NAME]/my_libero_smolvla \
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
    --peft.r=32 \
    --wandb.enable=true \
    --wandb.project=[WANDB_PROJ_NAME] \
    --wandb.entity=[WANDB_USER_NAME]
```

Use the following command for eval:

```python
lerobot-eval \
    --policy.path=[CKPT_PATH] \
    --policy.repo_id=[HUGGINGFACE_USER_NAME]/my_libero_smolvla \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=2 \
    --eval.n_episodes=20 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --output_dir=[OUTPUT_DIR] \
    --rename_map='{
    "observation.images.image2": "observation.images.wrist"
    }'
```