'''
### joint回放
```sh
python /home/yichu/yichu_work/data_collection/replay_joint.py \
  --server 192.168.36.246:50051 \
  --data /home/yichu/yichu_work/datasets/joint_record/episode_0000/joint_data.npz



python /home/yichu/yichu_work/data_collection/replay_joint.py \
  --server 192.168.36.116:50051 \
  --data /path/to/joint_data.npz --speed 0.5
'''

import time
import json
import signal
import sys
from pathlib import Path
from typing import Annotated, Optional
import typer
import numpy as np
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlModeParam, ManipulatorControlMode,
    JointPositions,
)


def load_joint_data(npz_path: str):
    data = np.load(npz_path)
    timestamps = data["timestamps"]
    left_pos = data["left_arm_position"]
    right_pos = data["right_arm_position"]
    try:
        joint_names = data["joint_names"]
    except KeyError:
        joint_names = None
    meta_path = Path(npz_path).parent / "episode.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    print(f"Loaded {len(timestamps)} frames from {npz_path}")
    print(f"  duration: {meta.get('duration', '?')}s, avg_hz: {meta.get('avg_hz', '?')}")
    print(f"  task: {meta.get('task', '?')}, robot: {meta.get('robot_model', '?')}")
    return timestamps, left_pos, right_pos, meta


def main(
    server: Annotated[str, typer.Option(help="Robot server address")] = "localhost:50051",
    data: Annotated[str, typer.Option(help="Path to joint_data.npz file")] = ...,
    speed: Annotated[float, typer.Option(help="Playback speed multiplier")] = 1.0,
    loop: Annotated[bool, typer.Option(help="Loop replay indefinitely")] = False,
):
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    timestamps, left_pos, right_pos, meta = load_joint_data(data)
    n = len(timestamps)
    if n == 0:
        print("No data to replay")
        return
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"Connected: {model}")
    #robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK), timeout=10)
    for attempt in range(3):
        r = robot.robot_control.set_manipulator_control_mode(
            ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS),
            timeout=10,
        )
        if r.is_success:
            break
        print(f"set_manipulator_control_mode attempt {attempt + 1}/3 failed")
        time.sleep(1)
    loop_count = 0
    while True:
        loop_count += 1
        if loop:
            print(f"\n=== Replay loop {loop_count} ===")
        data_start_ts = timestamps[0]
        replay_start = time.time()
        cmd_count = 0
        for i in range(n):
            target_real = replay_start + (timestamps[i] - data_start_ts) / speed
            now = time.time()
            if target_real > now:
                while time.time() < target_real:
                    pass
            try:
                robot.left_arm.set_joint_positions(JointPositions(positions=left_pos[i].tolist()))
                robot.right_arm.set_joint_positions(JointPositions(positions=right_pos[i].tolist()))
                cmd_count += 1
            except Exception as e:
                print(f"  frame {i} send failed: {e}")
            if i % max(1, n // 100) == 0 or i == n - 1:
                elapsed = time.time() - replay_start
                pct = (i + 1) / n * 100
                rate = cmd_count / elapsed if elapsed > 0 else 0
                print(f"\r  {pct:.0f}% ({i+1}/{n}) | {elapsed:.1f}s | {rate:.0f} Hz", end="", flush=True)
        elapsed = time.time() - replay_start
        rate = cmd_count / elapsed if elapsed > 0 else 0
        print(f"\nReplay completed: {cmd_count} frames in {elapsed:.2f}s ({rate:.0f} Hz)")
        if not loop:
            break
        print("Looping... (Ctrl+C to stop)")


if __name__ == "__main__":
    typer.run(main)
