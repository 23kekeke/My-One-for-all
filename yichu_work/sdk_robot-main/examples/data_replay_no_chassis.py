"""
Data Replay Example (No Chassis)

Same as data_replay_example.py but with chassis control disabled.
"""
import json
import time
import math
from typing import Annotated, Literal
from pathlib import Path
import typer
import signal
import sys
import threading
from x2robot import connect
from x2robot.sdk import ManipulatorControlMode, ManipulatorControlModeParam, JointPositions
from x2robot.sdk import GripperPosition, LiftPosition, HeadPose
from x2robot import geometry_msgs
from x2robot.sdk import RobotModeParam, RobotWorkMode
import numpy as np


def signal_handler(sig, frame):
    print("\n\nReceived interrupt signal, stopping...")
    sys.exit(0)


def load_episode_data(episode_path: str) -> dict:
    episode_file = Path(episode_path) / "episode.json"
    if not episode_file.exists():
        raise FileNotFoundError(f"File not found: {episode_file}")
    print(f"Loading data: {episode_file}")
    with open(episode_file, 'r') as f:
        data = json.load(f)
    print(f"  - Episode ID: {data['episode_id']}")
    print(f"  - Task: {data['task']}")
    print(f"  - Total frames: {data['num_frames']}")
    print(f"  - Duration: {data['duration']:.2f}s")
    return data


def filter_nan_values(joint_positions: list) -> list:
    filtered = []
    for pos in joint_positions:
        if pos is None or (isinstance(pos, float) and math.isnan(pos)):
            continue
        filtered.append(pos)
    return filtered


def replay_by_joint_positions(robot, episode_data: dict, playback_speed: float = 1.0):
    print("\n" + "="*60)
    print("Replay mode: Joint position control (chassis disabled)")
    print("="*60)

    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))

    robot_model = robot.get_robot_model()

    # 先释放急停
    robot.robot_control.recover_emergency_stop()
    time.sleep(0.5)

    print("Setting control mode to joint position control...")

    for attempt in range(3):
        try:
            mode_param = ManipulatorControlModeParam(
                mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS
            )
            result = robot.robot_control.set_manipulator_control_mode(mode_param)
            if result.is_success:
                break
            print(f"  set_manipulator_control_mode attempt {attempt + 1}/3: is_success={result.is_success}, error={result.error_message}")
        except Exception as e:
            print(f"  set_manipulator_control_mode attempt {attempt + 1}/3 failed: {e}")
        time.sleep(1)
    else:
        print("  Failed to set control mode after 3 attempts")
        return

    print("  Control mode set successfully")

    # Chassis disabled
    chassis_controller = None
    print("  Chassis control: disabled")

    # Enable motors / release brakes via homing
    print("Homing robot to enable motors...")
    robot.robot_control.recover_emergency_stop()
    homing_result = robot.robot_control.homing()
    if not homing_result.is_success:
        print(f"  Homing failed: {homing_result.error_message}")
    else:
        print("  Homing in progress (5s)...")
        time.sleep(5.0)

    # Homing may reset control mode; re-set it
    print("Re-setting control mode after homing...")
    mode_param = ManipulatorControlModeParam(
        mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS
    )
    result = robot.robot_control.set_manipulator_control_mode(mode_param)
    if not result.is_success:
        print(f"  Warning: Re-setting control mode failed: {result.error_message}")

    frames = episode_data['frames']
    total_frames = len(frames)

    # === DEBUG: send frame 0 once and verify ===
    print("\n--- DEBUG: Sending frame 0 only once ---")
    f0_obs = frames[0]['observation']
    test_left = filter_nan_values(f0_obs['left_arm_joint_states']['positions'])
    test_right = filter_nan_values(f0_obs['right_arm_joint_states']['positions'])

    # Read before
    lj_before = robot.left_arm.get_joint_states()
    rj_before = robot.right_arm.get_joint_states()
    print(f"  Before - left:  {[f'{p:.4f}' for p in lj_before.position]}")
    print(f"  Before - right: {[f'{p:.4f}' for p in rj_before.position]}")

    # Send
    print(f"  Sending left:  {[f'{p:.4f}' for p in test_left]}")
    print(f"  Sending right: {[f'{p:.4f}' for p in test_right]}")
    rl = robot.left_arm.set_joint_positions(JointPositions(positions=test_left))
    rr = robot.right_arm.set_joint_positions(JointPositions(positions=test_right))
    print(f"  Left result:  success={rl.is_success}")
    print(f"  Right result: success={rr.is_success}")

    time.sleep(3)

    # Read after
    lj_after = robot.left_arm.get_joint_states()
    rj_after = robot.right_arm.get_joint_states()
    print(f"  After  - left:  {[f'{p:.4f}' for p in lj_after.position]}")
    print(f"  After  - right: {[f'{p:.4f}' for p in rj_after.position]}")

    left_moved = any(abs(a - b) > 0.01 for a, b in zip(lj_before.position, lj_after.position))
    right_moved = any(abs(a - b) > 0.01 for a, b in zip(rj_before.position, rj_after.position))
    print(f"  Left MOVED: {left_moved}")
    print(f"  Right MOVED: {right_moved}")
    if left_moved or right_moved:
        print("  ✓ set_joint_positions works with data values. Continuing replay...")
    else:
        print("  ✗ set_joint_positions did NOT move the robot! Check control mode.")
    print("--- END DEBUG ---\n")

    print(f"\nStarting replay of {total_frames} frames...")
    print("Press Ctrl+C to stop replay\n")

    start_time = time.time()

    for i, frame in enumerate(frames):
        try:
            t0 = time.time()
            observation = frame.get('observation', {})

            left_arm_joint_states = observation.get('left_arm_joint_states')
            if left_arm_joint_states and 'positions' in left_arm_joint_states:
                left_arm_positions = filter_nan_values(left_arm_joint_states['positions'])
                if left_arm_positions:
                    left_joint_msg = JointPositions(positions=left_arm_positions)
                    t_left = time.time()
                    left_result = robot.left_arm.set_joint_positions(left_joint_msg)
                    dt_left = time.time() - t_left
                    if not left_result.is_success:
                        print(f"Warning: Left arm control failed - {left_result.error_message}")

            right_arm_joint_states = observation.get('right_arm_joint_states')
            if right_arm_joint_states and 'positions' in right_arm_joint_states:
                right_arm_positions = filter_nan_values(right_arm_joint_states['positions'])
                if right_arm_positions:
                    right_joint_msg = JointPositions(positions=right_arm_positions)
                    t_right = time.time()
                    right_result = robot.right_arm.set_joint_positions(right_joint_msg)
                    dt_right = time.time() - t_right
                    if not right_result.is_success:
                        print(f"Warning: Right arm control failed - {right_result.error_message}")

            if (robot_model == "quanta_x2"):
                left_gripper_position = observation.get('left_gripper_position')
                right_gripper_position = observation.get('right_gripper_position')
                if left_gripper_position and right_gripper_position:
                    try:
                        position = left_gripper_position['position']
                        robot.left_gripper.set_position(GripperPosition(position=position))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Left gripper control failed - {e}")
                    try:
                        position = right_gripper_position['position']
                        robot.right_gripper.set_position(GripperPosition(position=position))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Right gripper control failed - {e}")
            else:
                left_gripper_joint_states = observation.get('left_gripper_joint_states')
                if left_gripper_joint_states and 'positions' in left_gripper_joint_states:
                    left_gripper_positions = filter_nan_values(left_gripper_joint_states['positions'])
                    if left_gripper_positions:
                        try:
                            robot.left_gripper.set_position(GripperPosition(position=left_gripper_positions[0]))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Left gripper control failed - {e}")

                right_gripper_joint_states = observation.get('right_gripper_joint_states')
                if right_gripper_joint_states and 'positions' in right_gripper_joint_states:
                    right_gripper_positions = filter_nan_values(right_gripper_joint_states['positions'])
                    if right_gripper_positions:
                        try:
                            robot.right_gripper.set_position(GripperPosition(position=right_gripper_positions[0]))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Right gripper control failed - {e}")

            if robot_model == "quanta_x1":
                lift_joint_states = observation.get('lift_joint_states')
                if lift_joint_states and 'positions' in lift_joint_states:
                    lift_positions = filter_nan_values(lift_joint_states['positions'])
                    if lift_positions:
                        try:
                            robot.lift.set_lift_position(LiftPosition(position=lift_positions[0]))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Lift control failed - {e}")
            elif robot_model == "quanta_x2":
                waist_joint_states = observation.get('waist_joint_states')
                if waist_joint_states and 'positions' in waist_joint_states:
                    waist_positions = filter_nan_values(waist_joint_states['positions'])
                    if waist_positions:
                        try:
                            robot.waist.set_joint_positions(JointPositions(positions=waist_positions))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Waist control failed - {e}")
            elif robot_model == "desktop":
                pass

            head_joint_states = observation.get('head_joint_states')
            if head_joint_states and 'positions' in head_joint_states:
                head_positions = filter_nan_values(head_joint_states['positions'])
                if len(head_positions) >= 2:
                    try:
                        robot.head.set_pose(HeadPose(pitch=head_positions[0], yaw=head_positions[1]))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Head control failed - {e}")
                elif len(head_positions) == 1:
                    try:
                        robot.head.set_pose(HeadPose(pitch=0, yaw=head_positions[0]))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Head control failed - {e}")

            frame_dt = time.time() - t0
            if i < 20:
                print(f"Frame {i:3d}: dt_frame={frame_dt*1000:.1f}ms dt_left={dt_left*1000:.1f}ms dt_right={dt_right*1000:.1f}ms")

            progress = (i + 1) / total_frames * 100
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% ({i+1}/{total_frames}) | "
                      f"Time: {elapsed:.2f}s", end='\r')

            if i < total_frames - 1:
                next_timestamp = frames[i + 1]['timestamp']
                current_timestamp = frame['timestamp']
                sleep_time = (next_timestamp - current_timestamp) / playback_speed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nReplay interrupted by user")
            break
        except Exception as e:
            print(f"\nWarning: Frame {i} processing failed: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n\n  Replay completed! Time: {elapsed:.2f}s, Number of frames replayed: {i+1}")


def replay_by_end_pose(robot, episode_data: dict, playback_speed: float = 1.0):
    print("\n" + "="*60)
    print("Replay mode: End pose control (chassis disabled)")
    print("="*60)

    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))

    robot_model = robot.get_robot_model()

    print("Setting control mode to end pose control...")
    mode_param = ManipulatorControlModeParam(
        mode=ManipulatorControlMode.MANIPULATOR_END_POSE
    )
    result = robot.robot_control.set_manipulator_control_mode(mode_param)

    if not result.is_success:
        print(f"  Failed to set control mode: {result.error_message}")
        return

    print("  Control mode set successfully")
    # Chassis disabled
    print("  Chassis control: disabled")

    # Enable motors / release brakes via homing
    print("Homing robot to enable motors...")
    robot.robot_control.recover_emergency_stop()
    homing_result = robot.robot_control.homing()
    if not homing_result.is_success:
        print(f"  Homing failed: {homing_result.error_message}")
    else:
        print("  Homing in progress (5s)...")
        time.sleep(5.0)

    # Homing may reset control mode; re-set it
    print("Re-setting control mode after homing...")
    mode_param = ManipulatorControlModeParam(
        mode=ManipulatorControlMode.MANIPULATOR_END_POSE
    )
    result = robot.robot_control.set_manipulator_control_mode(mode_param)
    if not result.is_success:
        print(f"  Warning: Re-setting control mode failed: {result.error_message}")

    frames = episode_data['frames']
    total_frames = len(frames)

    print(f"\nStarting replay of {total_frames} frames...")
    print("Press Ctrl+C to stop replay\n")

    def is_valid_pose(pose_data):
        if not pose_data:
            return False
        pos = pose_data['position']
        ori = pose_data['orientation']
        return not (pos['x'] == 0 and pos['y'] == 0 and pos['z'] == 0 and
                   ori['x'] == 0 and ori['y'] == 0 and ori['z'] == 0 and ori['w'] == 1)

    start_time = time.time()
    valid_frames = 0

    for i, frame in enumerate(frames):
        try:
            observation = frame.get('observation', {})

            if robot_model == "quanta_x1":
                lift_joint_states = observation.get('lift_joint_states')
                if lift_joint_states and 'positions' in lift_joint_states:
                    lift_positions = filter_nan_values(lift_joint_states['positions'])
                    if lift_positions:
                        try:
                            robot.lift.set_lift_position(LiftPosition(position=lift_positions[0]))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Lift control failed - {e}")
            elif robot_model == "quanta_x2":
                waist_end_pose = observation.get('waist_end_pose')
                if waist_end_pose and is_valid_pose(waist_end_pose):
                    waist_position = geometry_msgs.Point(
                        x=waist_end_pose['position']['x'],
                        y=waist_end_pose['position']['y'],
                        z=waist_end_pose['position']['z']
                    )
                    waist_orientation = geometry_msgs.Quaternion(
                        x=waist_end_pose['orientation']['x'],
                        y=waist_end_pose['orientation']['y'],
                        z=waist_end_pose['orientation']['z'],
                        w=waist_end_pose['orientation']['w']
                    )
                    waist_pose_msg = geometry_msgs.Pose(
                        position=waist_position,
                        orientation=waist_orientation
                    )
                    waist_result = robot.waist.set_end_pose(waist_pose_msg)
                    if not waist_result.is_success:
                        if i == 0:
                            print(f"Warning: Waist end pose control failed - {waist_result.error_message}")
            elif robot_model == "desktop":
                pass

            left_end_pose = frame['observation'].get('left_arm_end_pose')
            right_end_pose = frame['observation'].get('right_arm_end_pose')

            if left_end_pose and is_valid_pose(left_end_pose):
                left_position = geometry_msgs.Point(
                    x=left_end_pose['position']['x'],
                    y=left_end_pose['position']['y'],
                    z=left_end_pose['position']['z']
                )
                left_orientation = geometry_msgs.Quaternion(
                    x=left_end_pose['orientation']['x'],
                    y=left_end_pose['orientation']['y'],
                    z=left_end_pose['orientation']['z'],
                    w=left_end_pose['orientation']['w']
                )
                left_pose_msg = geometry_msgs.Pose(
                    position=left_position,
                    orientation=left_orientation
                )
                left_result = robot.left_arm.set_end_pose(left_pose_msg)
                if not left_result.is_success:
                    print(f"Warning: Left arm end pose control failed - {left_result.error_message}")

            if right_end_pose and is_valid_pose(right_end_pose):
                right_position = geometry_msgs.Point(
                    x=right_end_pose['position']['x'],
                    y=right_end_pose['position']['y'],
                    z=right_end_pose['position']['z']
                )
                right_orientation = geometry_msgs.Quaternion(
                    x=right_end_pose['orientation']['x'],
                    y=right_end_pose['orientation']['y'],
                    z=right_end_pose['orientation']['z'],
                    w=right_end_pose['orientation']['w']
                )
                right_pose_msg = geometry_msgs.Pose(
                    position=right_position,
                    orientation=right_orientation
                )
                right_result = robot.right_arm.set_end_pose(right_pose_msg)
                if not right_result.is_success:
                    print(f"Warning: Right arm end pose control failed - {right_result.error_message}")
                valid_frames += 1

            if (robot_model == "quanta_x2"):
                left_gripper_position = observation.get('left_gripper_position')
                right_gripper_position = observation.get('right_gripper_position')
                if left_gripper_position and right_gripper_position:
                    try:
                        position = left_gripper_position['position']
                        robot.left_gripper.set_position(GripperPosition(position=position))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Left gripper control failed - {e}")
                    try:
                        position = right_gripper_position['position']
                        robot.right_gripper.set_position(GripperPosition(position=position))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Right gripper control failed - {e}")
            else:
                left_gripper_joint_states = observation.get('left_gripper_joint_states')
                if left_gripper_joint_states and 'positions' in left_gripper_joint_states:
                    left_gripper_positions = filter_nan_values(left_gripper_joint_states['positions'])
                    if left_gripper_positions:
                        try:
                            robot.left_gripper.set_position(GripperPosition(position=left_gripper_positions[0]))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Left gripper control failed - {e}")

                right_gripper_joint_states = observation.get('right_gripper_joint_states')
                if right_gripper_joint_states and 'positions' in right_gripper_joint_states:
                    right_gripper_positions = filter_nan_values(right_gripper_joint_states['positions'])
                    if right_gripper_positions:
                        try:
                            robot.right_gripper.set_position(GripperPosition(position=right_gripper_positions[0]))
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Right gripper control failed - {e}")

            head_joint_states = observation.get('head_joint_states')
            if head_joint_states and 'positions' in head_joint_states:
                head_positions = filter_nan_values(head_joint_states['positions'])
                if len(head_positions) >= 2:
                    try:
                        robot.head.set_pose(HeadPose(pitch=head_positions[0], yaw=head_positions[1]))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Head control failed - {e}")
                elif len(head_positions) == 1:
                    try:
                        robot.head.set_pose(HeadPose(pitch=0, yaw=head_positions[0]))
                    except Exception as e:
                        if i == 0:
                            print(f"Warning: Head control failed - {e}")

            progress = (i + 1) / total_frames * 100
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% ({i+1}/{total_frames}) | "
                      f"Valid frames: {valid_frames} | Time: {elapsed:.2f}s", end='\r')

            if i < total_frames - 1:
                next_timestamp = frames[i + 1]['timestamp']
                current_timestamp = frame['timestamp']
                sleep_time = (next_timestamp - current_timestamp) / playback_speed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nReplay interrupted by user")
            break
        except Exception as e:
            print(f"\nWarning: Frame {i} processing failed: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n\n  Replay completed! Time: {elapsed:.2f}s, Number of frames replayed: {i+1}")


def main(
    episode_dir: Annotated[str, typer.Argument(help="Episode directory path")],
    server: Annotated[str, typer.Option(help="Robot server address")] = "localhost:50051",
    mode: Annotated[Literal["joint_pos", "end_pose"], typer.Option(help="Replay mode")] = "joint_pos",
    speed: Annotated[float, typer.Option(help="Playback speed")] = 1.0,
):
    signal.signal(signal.SIGINT, signal_handler)

    try:
        episode_data = load_episode_data(episode_dir)
    except Exception as e:
        print(f"  Failed to load data: {e}")
        sys.exit(1)

    print(f"\nConnecting to robot {server}...")
    try:
        robot = connect(f"x2://{server}")
        print("  Robot connected successfully")
        print(f"  Robot model: {robot.get_robot_model()}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    robot_model = robot.get_robot_model()
    print(f"\nReplay configuration:")
    print(f"  - Mode: {mode}")
    print(f"  - Speed: {speed}x")
    print(f"  - Number of frames: {episode_data['num_frames']}")
    print(f"  - Chassis: disabled")

    record_data_model = episode_data['model']
    if (record_data_model != robot_model):
        print(f"  Record data model {record_data_model} does not match robot model {robot_model}")
        sys.exit(1)

    input("\nPress Enter to start replay...")

    try:
        if mode == "joint_pos":
            replay_by_joint_positions(robot, episode_data, speed)
        elif mode == "end_pose":
            replay_by_end_pose(robot, episode_data, speed)
        else:
            print(f"  Unknown replay mode: {mode}")
            sys.exit(1)
    except Exception as e:
        print(f"\n  Error during replay: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nReplay completed")


if __name__ == "__main__":
    typer.run(main)
