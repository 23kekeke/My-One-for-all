'''
### joint回放
```sh
python /home/yichu/yichu_work/data_collection/replay_joint

.py \
  --server 192.168.36.246:50051 \
  --data /home/yichu/yichu_work/datasets/joint_record/episode_0000/joint_data.npz
  --speed 0.1 \
  --skip-work-mode


python /home/yichu/yichu_work/data_collection/replay_joint.py \
  --server 192.168.36.246:50051 \
  --data /home/yichu/yichu_work/datasets/joint_record/episode_0000/joint_data.npz \
  --speed 0.5
```
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
    GripperPosition,
)


def _load_optional_array(data, key: str):
    """Return data[key] if present, else None."""
    try:
        return data[key]
    except KeyError:
        return None


def _as_scalar_1d_value(arr, i: int) -> Optional[float]:
    """Read frame i from a gripper array shaped (N,), (N,1), etc."""
    if arr is None:
        return None
    v = np.asarray(arr[i]).reshape(-1)
    if v.size == 0:
        return None
    return float(v[0])


def load_joint_data(npz_path: str):
    data = np.load(npz_path)
    timestamps = data["timestamps"]
    left_pos = data["left_arm_position"]
    right_pos = data["right_arm_position"]

    left_gripper_pos = _load_optional_array(data, "left_gripper_position")
    right_gripper_pos = _load_optional_array(data, "right_gripper_position")

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
    print(f"  left_arm_position: {left_pos.shape}")
    print(f"  right_arm_position: {right_pos.shape}")
    print(f"  left_gripper_position: {None if left_gripper_pos is None else left_gripper_pos.shape}")
    print(f"  right_gripper_position: {None if right_gripper_pos is None else right_gripper_pos.shape}")

    return timestamps, left_pos, right_pos, left_gripper_pos, right_gripper_pos, meta


def main(
    server: Annotated[str, typer.Option(help="Robot server address")] = "localhost:50051",
    data: Annotated[str, typer.Option(help="Path to joint_data.npz file")] = ...,
    speed: Annotated[float, typer.Option(help="Playback speed multiplier")] = 1.0,
    loop: Annotated[bool, typer.Option(help="Loop replay indefinitely")] = False,
    replay_gripper: Annotated[bool, typer.Option(help="Replay gripper positions if available")] = True,
    gripper_eps: Annotated[float, typer.Option(help="Only send gripper command when target changes more than this. Use 0 for every frame.")] = 0.002,
):
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    timestamps, left_pos, right_pos, left_gripper_pos, right_gripper_pos, meta = load_joint_data(data)
    n = len(timestamps)
    if n == 0:
        print("No data to replay")
        return

    has_left_gripper = left_gripper_pos is not None and len(left_gripper_pos) == n
    has_right_gripper = right_gripper_pos is not None and len(right_gripper_pos) == n
    if replay_gripper:
        if not has_left_gripper:
            print("warning: no valid left_gripper_position found, left gripper replay disabled")
        if not has_right_gripper:
            print("warning: no valid right_gripper_position found, right gripper replay disabled")
    else:
        print("Gripper replay disabled by --no-replay-gripper")

    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"Connected: {model}")

    # Keep the original replay setup exactly the same.
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
        gripper_cmd_count = 0
        last_left_gripper = None
        last_right_gripper = None

        for i in range(n):
            # Keep the original timestamp replay logic exactly the same.
            target_real = replay_start + (timestamps[i] - data_start_ts) / speed
            now = time.time()
            if target_real > now:
                while time.time() < target_real:
                    pass

            # Keep the original arm replay commands exactly the same.
            try:
                robot.left_arm.set_joint_positions(JointPositions(positions=left_pos[i].tolist()))
                robot.right_arm.set_joint_positions(JointPositions(positions=right_pos[i].tolist()))
                cmd_count += 1
            except Exception as e:
                print(f"  frame {i} arm send failed: {e}")

            # Minimal addition: gripper replay uses the confirmed GripperPosition(position=...).
            if replay_gripper:
                if has_left_gripper:
                    left_g = _as_scalar_1d_value(left_gripper_pos, i)
                    if left_g is not None and (
                        last_left_gripper is None or abs(left_g - last_left_gripper) > gripper_eps
                    ):
                        try:
                            robot.left_gripper.set_position(GripperPosition(position=left_g))
                            last_left_gripper = left_g
                            gripper_cmd_count += 1
                        except Exception as e:
                            print(f"  frame {i} left gripper send failed: {e}")

                if has_right_gripper:
                    right_g = _as_scalar_1d_value(right_gripper_pos, i)
                    if right_g is not None and (
                        last_right_gripper is None or abs(right_g - last_right_gripper) > gripper_eps
                    ):
                        try:
                            robot.right_gripper.set_position(GripperPosition(position=right_g))
                            last_right_gripper = right_g
                            gripper_cmd_count += 1
                        except Exception as e:
                            print(f"  frame {i} right gripper send failed: {e}")

            if i % max(1, n // 100) == 0 or i == n - 1:
                elapsed = time.time() - replay_start
                pct = (i + 1) / n * 100
                rate = cmd_count / elapsed if elapsed > 0 else 0
                print(f"\r  {pct:.0f}% ({i+1}/{n}) | {elapsed:.1f}s | {rate:.0f} Hz | gripper_cmds={gripper_cmd_count}", end="", flush=True)

        elapsed = time.time() - replay_start
        rate = cmd_count / elapsed if elapsed > 0 else 0
        print(f"\nReplay completed: {cmd_count} frames in {elapsed:.2f}s ({rate:.0f} Hz), gripper_cmds={gripper_cmd_count}")

        if not loop:
            break
        print("Looping... (Ctrl+C to stop)")


if __name__ == "__main__":
    typer.run(main)
