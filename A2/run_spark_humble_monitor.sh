#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yichu/A2"
IMAGE="${A2_DOCKER_IMAGE:-a2-spark-humble:20260731}"
PROFILE="$ROOT/config/spark_ros_dds_configuration.xml"
MONITOR_SECONDS="${1:-30}"

test -f "$PROFILE" || {
  echo "错误：找不到 $PROFILE" >&2
  exit 1
}
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi
"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "错误：找不到镜像 $IMAGE，请先运行 build_spark_humble_container.sh" >&2
  exit 1
}
mkdir -p "$ROOT/logs"

exec "${DOCKER[@]}" run --rm --interactive --tty \
  --name a2-spark-humble-monitor \
  --user "$(id -u):$(id -g)" \
  --network host \
  --ipc host \
  --volume "$ROOT:/workspace" \
  --env ROS_DOMAIN_ID=232 \
  --env ROS_LOCALHOST_ONLY=0 \
  --env ROS_LOG_DIR=/workspace/logs \
  --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  --env FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/spark_ros_dds_configuration.xml \
  "$IMAGE" \
  python3 /workspace/a2_ros2_live_episode_collector.py \
    --output-root /workspace/data/raw \
    --monitor-seconds "$MONITOR_SECONDS"
