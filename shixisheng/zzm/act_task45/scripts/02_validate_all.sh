#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

while IFS= read -r group || [[ -n "$group" ]]; do
  group="${group%%#*}"
  group="$(echo "$group" | xargs)"
  [[ -z "$group" ]] && continue

  python "$PROJECT_DIR/tools/validate_lerobot_dataset.py" \
    --root "$CONVERTED_BASE/${group}_raw14" \
    --repo-id "dongguan/${group}_raw14"
done < "$GROUP_LIST"

