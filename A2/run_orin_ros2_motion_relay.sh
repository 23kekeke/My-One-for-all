#!/usr/bin/env bash
set -eo pipefail

ROOT="/agibot/data/a2_ros_relay"

source /opt/ros/humble/setup.bash
set -u

# Use fixed acquisition-network settings instead of inheriting possibly stale
# values from an interactive shell.
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/v0/entry/bin/cfg/ros_dds_configuration.xml

exec /usr/bin/python3 "$ROOT/a2_orin_ros2_motion_relay.py"
