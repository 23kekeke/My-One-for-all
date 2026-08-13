#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

python "$PROJECT_DIR/tools/audit_raw_dataset.py" "$RAW_DATASET"

