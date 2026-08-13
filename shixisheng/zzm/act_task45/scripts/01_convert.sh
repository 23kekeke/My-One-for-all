#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/01_convert_all.sh" "$@"

$CONVERTED_DATASET" ]]; then
  echo "Output already exists; refusing to overwrite: $CONVERTED_DATASET" >&2
  echo "Move it to a backup path or choose another CONVERTED_DATASET in config.sh." >&2
  exit 1
fi

python "$PROJECT_DIR/tools/convert_to_lerobot_async.py" \
  --input-dir "$RAW_DATASET" \
  --output-dir "$CONVERTED_DATASET" \
  --repo-id "$DATASET_REPO_ID" \
  --robot-type quanta_x1_raw_joints \
  --fps 30 \
  --use-videos \
  --processes 4

