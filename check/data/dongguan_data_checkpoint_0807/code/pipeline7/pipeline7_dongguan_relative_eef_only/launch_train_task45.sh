#!/usr/bin/env bash
# Launch pipeline7 task45 formal training with stdout/stderr tee'd to train.log.
#
# Usage:
#   ./launch_train_task45.sh
#   ./launch_train_task45.sh --resume-from-checkpoint
#   ./launch_train_task45.sh --no-use-wandb
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${ROOT}/isaacGr00t/.venv/bin/python"
ENTRY="${SCRIPT_DIR}/dongguan_eef_only_finetune_entry.py"
MODALITY="${SCRIPT_DIR}/dongguan_eef_only_relative_modality.py"
BASE_MODEL="${ROOT}/GR00T-N1.7-3B"
DATASET="/data/dongguan_data_checkpoint_0807/lerobot_p7/task45"
OUTPUT_ROOT="/data/dongguan_data_checkpoint_0807/output_p7"
RUN_NAME="dongguan_task45_eef_only_relative_qlora_r32_vit_r32_bs2_ga4"

export GR00T_COSMOS_MODEL_PATH="${ROOT}/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PIPELINE2_QLORA=1
export GROOT_PATCH_MISTRAL=1
export GROOT_HF_LOCAL_FIRST=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/isaacGr00t:${ROOT}/pipeline2:${ROOT}/pipeline3_biman:${SCRIPT_DIR}:${PYTHONPATH:-}"

USE_WANDB=1
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-use-wandb)
      USE_WANDB=0
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "[launch_train_task45] dataset=$DATASET"
echo "[launch_train_task45] output=$OUTPUT_DIR"
echo "[launch_train_task45] log -> $OUTPUT_DIR/train.log"

CMD=(
  "$PYTHON" "$ENTRY"
  --base-model-path "$BASE_MODEL"
  --dataset-path "$DATASET"
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path "$MODALITY"
  --output-dir "$OUTPUT_DIR"
  --num-gpus 1
  --max-steps 15000
  --save-steps 1000
  --save-total-limit 7
  --global-batch-size 2
  --gradient-accumulation-steps 4
  --dataloader-num-workers 2
  --episode-sampling-rate 1.0
  --num-shards-per-epoch 211
  --shard-size 128
  --learning-rate 1e-4
  --warmup-ratio 0.05
  --weight-decay 1e-5
  --no-tune-llm
  --no-tune-visual
  --tune-projector
  --tune-diffusion-model
  --state-dropout-prob 0.2
  --wandb-project dongguan-pipeline7
)

if [[ "$USE_WANDB" -eq 1 ]]; then
  CMD+=(--use-wandb)
fi

CMD+=("${EXTRA_ARGS[@]}")

printf '%q ' "${CMD[@]}" > "$OUTPUT_DIR/launch_command.sh"
echo >> "$OUTPUT_DIR/launch_command.sh"

"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
