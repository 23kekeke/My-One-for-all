#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yichu/A2"
IMAGE="${A2_DOCKER_IMAGE:-a2-spark-humble:20260731}"
BASE_IMAGE="${A2_BASE_IMAGE:-ros:humble-ros-base-jammy}"

command -v docker >/dev/null || {
  echo "错误：Spark没有安装docker" >&2
  exit 1
}

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

"${DOCKER[@]}" build \
  --pull=false \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$ROOT/docker/Dockerfile.spark-humble" \
  --tag "$IMAGE" \
  "$ROOT"

echo "image=$IMAGE"
echo "base_image=$BASE_IMAGE"
