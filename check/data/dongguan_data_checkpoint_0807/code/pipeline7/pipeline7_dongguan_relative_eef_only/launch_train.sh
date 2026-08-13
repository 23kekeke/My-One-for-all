#!/usr/bin/env bash
# Launch pipeline7 formal training with stdout/stderr tee'd to train.log.
#
# Usage:
#   ./launch_train.sh multi345_v1
#   ./launch_train.sh multi345_v1 --resume-from-checkpoint
#   ./launch_train.sh multi345_v1 --use-wandb
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${ROOT}/isaacGr00t/.venv/bin/python"
ENTRY="${SCRIPT_DIR}/dongguan_eef_only_finetune_entry.py"
MODALITY="${SCRIPT_DIR}/dongguan_eef_only_relative_modality.py"
BASE_MODEL="${ROOT}/GR00T-N1.7-3B"
OUTPUT_ROOT="/data/dongguan_data_checkpoint_0807/output_p7"

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

RECIPE="${1:-multi345_v1}"
shift || true

case "$RECIPE" in
  multi345_v1)
    DATASET="/data/dongguan_data_checkpoint_0807/lerobot_p7/multi_345"
    RUN_NAME="dongguan_multi345_eef_only_relative_qlora_r32_vit_r32_bs2_ga4"
    EXTRA=(
      --max-steps 15000
      --save-steps 1000
      --save-total-limit 7
      --global-batch-size 2
      --gradient-accumulation-steps 4
      --dataloader-num-workers 2
      --episode-sampling-rate 1.0
      --num-shards-per-epoch 512
      --shard-size 128
      --learning-rate 1e-4
      --warmup-ratio 0.05
      --weight-decay 1e-5
      --state-dropout-prob 0.2
      --wandb-project dongguan-pipeline7
    )
    ;;
  *)
    echo "Unknown recipe: $RECIPE (use multi345_v1)" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "[launch_train] recipe=$RECIPE output=$OUTPUT_DIR"
echo "[launch_train] log -> $OUTPUT_DIR/train.log"

CMD=(
  "$PYTHON" "$ENTRY"
  --base-model-path "$BASE_MODEL"
  --dataset-path "$DATASET"
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path "$MODALITY"
  --output-dir "$OUTPUT_DIR"
  --num-gpus 1
  --no-tune-llm
  --no-tune-visual
  --tune-projector
  --tune-diffusion-model
  "${EXTRA[@]}"
  "$@"
)

printf '%q ' "${CMD[@]}" > "$OUTPUT_DIR/launch_command.sh"
echo >> "$OUTPUT_DIR/launch_command.sh"

"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
