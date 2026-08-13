#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

if [[ -e "$MERGED_DATASET" ]]; then
  echo "Merged output already exists; refusing to overwrite: $MERGED_DATASET" >&2
  exit 1
fi

repo_items=()
root_items=()
while IFS= read -r group || [[ -n "$group" ]]; do
  group="${group%%#*}"
  group="$(echo "$group" | xargs)"
  [[ -z "$group" ]] && continue
  root="$CONVERTED_BASE/${group}_raw14"
  [[ -d "$root" ]] || { echo "Converted dataset missing: $root" >&2; exit 1; }
  repo_items+=("'dongguan/${group}_raw14'")
  root_items+=("'$root'")
done < "$GROUP_LIST"

repo_list="[$(IFS=,; echo "${repo_items[*]}")]"
root_list="[$(IFS=,; echo "${root_items[*]}")]"

cd "$LEROBOT_REPO"
python -m lerobot.scripts.lerobot_edit_dataset \
  --new_repo_id "$MERGED_REPO_ID" \
  --new_root "$MERGED_DATASET" \
  --operation.type merge \
  --operation.repo_ids "$repo_list" \
  --operation.roots "$root_list"

