#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "用法: $0 RAW_BAG_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

ROOT="/home/yichu/A2"
EXTRACTOR="/home/yichu/yichu_work/data_collection/extract_a2_head_hands_h265_episode.py"
RAW_BAG="$(realpath "$1")"
EPISODE_NAME="$(basename "$RAW_BAG")"
OUTPUT_DIR="${2:-$ROOT/processed/$EPISODE_NAME}"

test -f "$RAW_BAG/metadata.yaml" || {
  echo "错误：找不到 $RAW_BAG/metadata.yaml" >&2
  exit 1
}
test -f "$EXTRACTOR" || {
  echo "错误：找不到 $EXTRACTOR" >&2
  exit 1
}
if [ -e "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "错误：输出目录非空：$OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

/usr/bin/python3 "$EXTRACTOR" "$RAW_BAG" "$OUTPUT_DIR"

/usr/bin/python3 \
  "$ROOT/convert_h265_episode_to_15fps.py" \
  "$OUTPUT_DIR" \
  --output-fps 15

/usr/bin/python3 \
  "$ROOT/resample_a2_motion_to_50hz.py" \
  "$RAW_BAG" \
  "$OUTPUT_DIR/signals_50hz.json"

(
  cd "$OUTPUT_DIR"
  sha256sum \
    head_camera.mp4 \
    left_hand_camera.mp4 \
    right_hand_camera.mp4 \
    episode.json \
    signals_50hz.json \
    > SHA256SUMS
)

echo "处理完成：$OUTPUT_DIR"
