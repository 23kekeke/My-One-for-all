"""
Minimal replay test: send one frame of data values and verify position changed.
"""
import json, time
import typer
from typing import Annotated
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlMode, ManipulatorControlModeParam,
    JointPositions
)


def main(
    episode_dir: Annotated[str, typer.Argument(help="Episode directory")],
    server: Annotated[str, typer.Option(help="Robot address")] = "localhost:50051",
):
    # Load first frame data
    with open(f"{episode_dir}/episode.json") as f:
        data = json.load(f)
    f0 = data['frames'][0]['observation']
    target_left = f0['left_arm_joint_states']['positions']
    target_right = f0['right_arm_joint_states']['positions']
    print(f"Frame 0 left arm:  {[f'{p:.4f}' for p in target_left]}")
    print(f"Frame 0 right arm: {[f'{p:.4f}' for p in target_right]}")

    # Connect
    print(f"\nConnecting to {server}...")
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"Model: {model}")

    # Set modes
    print("set_work_mode(SDK)...")
    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))

    print("set_manipulator_control_mode(JOINT_POSITIONS)...")
    r = robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )
    print(f"  success={r.is_success}")

    # Homing
    print("recover_emergency_stop...")
    robot.robot_control.recover_emergency_stop()

    print("homing (robot will move to home)...")
    r = robot.robot_control.homing()
    print(f"  success={r.is_success}")
    time.sleep(5)

    # Read current positions after homing
    lj0 = robot.left_arm.get_joint_states()
    rj0 = robot.right_arm.get_joint_states()
    print(f"\nAfter homing - left:  {[f'{p:.4f}' for p in lj0.position]}")
    print(f"After homing - right: {[f'{p:.4f}' for p in rj0.position]}")

    # Send frame 0 RIGHT arm positions
    print(f"\nSending frame 0 right arm positions...")
    r_result = robot.right_arm.set_joint_positions(JointPositions(positions=target_right))
    print(f"  result: success={r_result.is_success}")

    # Send frame 0 LEFT arm positions
    print(f"Sending frame 0 left arm positions...")
    l_result = robot.left_arm.set_joint_positions(JointPositions(positions=target_left))
    print(f"  result: success={l_result.is_success}")

    print("Waiting 5s for movement...")
    time.sleep(5)

    # Read positions to verify
    lj1 = robot.left_arm.get_joint_states()
    rj1 = robot.right_arm.get_joint_states()
    print(f"\nAfter sending - left:  {[f'{p:.4f}' for p in lj1.position]}")
    print(f"After sending - right: {[f'{p:.4f}' for p in rj1.position]}")

    # Check if anything changed
    left_changed = any(abs(a - b) > 0.001 for a, b in zip(lj0.position, lj1.position))
    right_changed = any(abs(a - b) > 0.001 for a, b in zip(rj0.position, rj1.position))
    print(f"\nLeft arm moved:  {left_changed}")
    print(f"Right arm moved: {right_changed}")

    if not left_changed and not right_changed:
        print("\nARMS DID NOT MOVE - testing with diagnostic +0.05 rad offset...")
        # Try diagnostic-style test
        orig = list(lj0.position)
        test = orig.copy()
        test[0] += 0.05
        print(f"  Sending left joint 0: {orig[0]:.4f} -> {test[0]:.4f}")
        r = robot.left_arm.set_joint_positions(JointPositions(positions=test))
        print(f"  result: success={r.is_success}")
        time.sleep(3)
        lj2 = robot.left_arm.get_joint_states()
        print(f"  joint0 now: {lj2.position[0]:.4f}")
        moved = abs(lj2.position[0] - orig[0]) > 0.01
        print(f"  Moved: {moved}")


if __name__ == "__main__":
    typer.run(main)
