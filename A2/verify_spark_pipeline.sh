#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yichu/A2"

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

/usr/bin/python3 - <<'PY'
import rclpy
import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

print("ROS Python imports: OK")
PY

/usr/bin/python3 -m py_compile \
  "$ROOT/a2_ros2_live_episode_collector.py" \
  "$ROOT/convert_h265_episode_to_15fps.py" \
  "$ROOT/make_spark_fastdds_profile.py" \
  "$ROOT/resample_a2_motion_to_50hz.py"

test -f \
  /home/yichu/yichu_work/data_collection/extract_a2_head_hands_h265_episode.py

echo "Spark A2 pipeline static checks: PASS"
