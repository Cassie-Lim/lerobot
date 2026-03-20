
# lerobot-eval \
#     --policy.path=outputs/train/2026-03-19/21-24-34_libero_smolvla/checkpoints/last/pretrained_model \
#     --policy.repo_id=cassielin0910/my_libero_smolvla \
#     --env.type=libero \
#     --env.task=libero_spatial \
#     --eval.batch_size=1 \
#     --eval.n_episodes=20 \
#     --policy.use_amp=false \
#     --policy.device=cuda \
#     --output_dir=logs/eval/rank16 \
#     --rename_map='{
#     "observation.images.image2": "observation.images.wrist"
#     }'


lerobot-eval \
    --policy.path=/coc/flash7/mlin365/lerobot/outputs/train/2026-03-19/21-40-31_libero_smolvla/checkpoints/last/pretrained_model \
    --policy.repo_id=cassielin0910/my_libero_smolvla \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=2 \
    --eval.n_episodes=20 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --output_dir=logs/eval/rank16 \
    --rename_map='{
    "observation.images.image2": "observation.images.wrist"
    }'

