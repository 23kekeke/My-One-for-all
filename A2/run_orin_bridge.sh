#!/usr/bin/env bash
set -eo pipefail

ROOT="/agibot/data/a2_grpc_bridge"

source /opt/ros/humble/setup.bash

# ROS 2 Humble setup scripts read a few optional variables before checking
# whether they exist, so nounset must only be enabled after sourcing ROS.
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/agibot/software/v0/entry/bin/cfg/ros_dds_configuration.xml}"

exec "$ROOT/.venv/bin/python" \
  "$ROOT/a2_orin_grpc_bridge.py" \
  --config "$ROOT/config/orin_bridge.json"
