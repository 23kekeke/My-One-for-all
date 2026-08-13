#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

RUN_DIR="$OUTPUT_ROOT/act_task45_smoke_pku"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
  exit 1
fi

# The PKU reference configuration uses TorchCodec for video loading.
python -c "import torchcodec" >/dev/null 2>&1 || {
  echo "torchcodec is required by the PKU-aligned training configuration." >&2
  echo "Install a torchcodec build compatible with the PyTorch version in $CONDA_ENV." >&2
  exit 1
}

cd "$LEROBOT_REPO"
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$CONVERTED_DATASET" \
  --dataset.image_transforms.enable=false \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=torchcodec \
  --dataset.streaming=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.vision_backbone=resnet18 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.n_vae_encoder_layers=4 \
  --policy.dropout=0.1 \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=0.0001 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.push_to_hub=false \
  --output_dir="$RUN_DIR" \
  --job_name=act_task45_smoke_pku \
  --seed=1000 \
  --cudnn_deterministic=false \
  --num_workers=4 \
  --batch_size=16 \
  --gradient_accumulation_steps=1 \
  --steps=2000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --eval_freq=0 \
  --use_policy_training_preset=true \
  --wandb.enable=false

