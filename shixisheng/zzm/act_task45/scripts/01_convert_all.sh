#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

if [[ ! -f "$GROUP_LIST" ]]; then
  echo "Dataset group list not found: $GROUP_LIST" >&2
  exit 1
fi

mkdir -p "$CONVERTED_BASE" "$PROJECT_DIR/reports"
ORDER_REPORT="$PROJECT_DIR/reports/conversion_order.tsv"
printf 'global_order\tgroup\tsource_episode\n' > "$ORDER_REPORT"
global_order=0

while IFS= read -r group || [[ -n "$group" ]]; do
  group="${group%%#*}"
  group="$(echo "$group" | xargs)"
  [[ -z "$group" ]] && continue

  input_dir="$RAW_BASE/$group"
  output_dir="$CONVERTED_BASE/${group}_raw14"
  repo_id="dongguan/${group}_raw14"

  if [[ ! -d "$input_dir" ]]; then
    echo "Input dataset not found: $input_dir" >&2
    exit 1
  fi
  if [[ ! -f "$input_dir/dataset_metadata.json" ]]; then
    echo "Metadata missing; repair before conversion: $input_dir/dataset_metadata.json" >&2
    exit 1
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Output already exists; refusing to overwrite: $output_dir" >&2
    exit 1
  fi

  python "$PROJECT_DIR/tools/check_dataset_structure.py" \
    --dataset "$input_dir" \
    --actual-only \
    --deep-json

  mapfile -t episode_names < <(
    find "$input_dir" -maxdepth 1 -type d -name 'episode_*' -printf '%f\n' | sort -V
  )
  if [[ ${#episode_names[@]} -eq 0 ]]; then
    echo "No episode directories found: $input_dir" >&2
    exit 1
  fi

  episode_ids=()
  for episode_name in "${episode_names[@]}"; do
    padded_id="${episode_name#episode_}"
    numeric_id=$((10#$padded_id))
    episode_ids+=("$numeric_id")
    printf '%d\t%s\t%d\n' "$global_order" "$group" "$numeric_id" >> "$ORDER_REPORT"
    global_order=$((global_order + 1))
  done
  episode_csv="$(IFS=,; echo "${episode_ids[*]}")"

  echo "Converting $group (${#episode_ids[@]} episodes, numeric order)"
  python "$PROJECT_DIR/tools/convert_to_lerobot_async.py" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --repo-id "$repo_id" \
    --robot-type quanta_x1_raw_joints \
    --episodes "$episode_csv" \
    --fps 30 \
    --use-videos \
    --processes 4
done < "$GROUP_LIST"

echo "All configured groups converted. Order report: $ORDER_REPORT"

