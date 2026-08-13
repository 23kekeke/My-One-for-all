#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -d "$LEROBOT_REPO/src/lerobot" ]]; then
  echo "LeRobot source tree not found: $LEROBOT_REPO" >&2
  exit 1
fi

export PYTHONPATH="$LEROBOT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

