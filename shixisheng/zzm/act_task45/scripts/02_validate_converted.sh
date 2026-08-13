#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

python "$PROJECT_DIR/tools/validate_lerobot_dataset.py" \
  --root "$CONVERTED_DATASET" \
  --repo-id "$DATASET_REPO_ID"

