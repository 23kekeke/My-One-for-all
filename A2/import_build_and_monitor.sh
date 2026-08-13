#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yichu/A2"
TAR="$ROOT/ros-humble-ros-base-jammy-arm64.tar"
SUM="$ROOT/ros-humble-ros-base-jammy-arm64.tar.sha256"
BASE_IMAGE="ros:humble-ros-base-jammy"
LOCAL_BASE="a2-local/ros:humble-base-arm64"
FINAL_IMAGE="a2-spark-humble:20260731"

cd "$ROOT"

test -f "$TAR" || {
  echo "ERROR: missing $TAR" >&2
  exit 1
}
test -f "$SUM" || {
  echo "ERROR: missing $SUM" >&2
  exit 1
}

echo "===== 1/6 SHA256 ====="
sha256sum -c "$SUM"

echo "===== 2/6 Docker load ====="
sudo docker load --input "$TAR"

echo "===== 3/6 Architecture ====="
platform="$(
  sudo docker image inspect \
    "$BASE_IMAGE" \
    --format '{{.Os}}/{{.Architecture}}'
)"
echo "platform=$platform"
test "$platform" = "linux/arm64" || {
  echo "ERROR: expected linux/arm64, found $platform" >&2
  exit 1
}

echo "===== 4/6 Local tag ====="
sudo docker tag "$BASE_IMAGE" "$LOCAL_BASE"

echo "===== 5/6 Build A2 Humble image ====="
export A2_BASE_IMAGE="$LOCAL_BASE"
export A2_DOCKER_IMAGE="$FINAL_IMAGE"
"$ROOT/build_spark_humble_container.sh"

echo "===== 6/6 Monitor 30 seconds ====="
"$ROOT/run_spark_humble_monitor.sh" 30
