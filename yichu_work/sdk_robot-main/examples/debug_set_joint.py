"""
Deep debug: check control mode and test set_joint_positions step by step.
"""
import json, time
import typer
from typing import Annotated
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlMode, ManipulatorControlModeParam,
    JointPositions, ChassisControlMode, ChassisControlModeParam,
    NavigationMode, NavigationModeParam,
)


def main(
    server: Annotated[str, typer.Option(help="Robot address")] = "localhost:50051",
):
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"Model: {model}")

    # Step 1: set work mode
    print("\n[1] set_work_mode(SDK)...")
    r = robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    print(f"    success={r.is_success}")

    # Step 2: set manipulator control mode
    print("\n[2] set_manipulator_control_mode(JOINT_POSITIONS)...")
    r = robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )
    print(f"    success={r.is_success}")

    # Step 3: verify mode
    print("\n[3] get_manipulator_control_mode...")
    mode = robot.robot_control.get_manipulator_control_mode()
    print(f"    mode={mode.mode}")

    # Step 4: recover emergency stop
    print("\n[4] recover_emergency_stop...")
    r = robot.robot_control.recover_emergency_stop()
    print(f"    success={r.is_success}")

    # Step 5: homing
    print("\n[5] homing...")
    r = robot.robot_control.homing()
    print(f"    success={r.is_success}")
    print("    waiting 6s...")
    time.sleep(6)

    # Step 6: check control mode after homing
    print("\n[6] get_manipulator_control_mode AFTER homing...")
    mode2 = robot.robot_control.get_manipulator_control_mode()
    print(f"    mode={mode2.mode}")

    # Step 7: re-set control mode (in case homing changed it)
    print("\n[7] Re-setting JOINT_POSITIONS mode...")
    r = robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )
    print(f"    success={r.is_success}")

    # Step 8: also try setting navigation mode (like sample does)
    print("\n[8] set_navigation_mode(BUILT_IN_NAVIGATION)...")
    try:
        r = robot.navigation.set_navigation_mode(
            NavigationModeParam(mode=NavigationMode.BUILT_IN_NAVIGATION)
        )
        print(f"    success={r.is_success}")
    except Exception as e:
        print(f"    N/A: {e}")

    # Step 9: read current joint positions
    print("\n[9] Current joint positions:")
    lj = robot.left_arm.get_joint_states()
    rj = robot.right_arm.get_joint_states()
    print(f"    left:  {[f'{p:.4f}' for p in lj.position]}")
    print(f"    right: {[f'{p:.4f}' for p in rj.position]}")

    # Step 10: try set_joint_positions with a BIG offset (0.3 rad)
    print("\n[10] set_joint_positions with +0.3 rad on left joint 0...")
    orig = list(lj.position)
    target = orig.copy()
    target[0] = orig[0] + 0.3
    print(f"    {orig[0]:.4f} -> {target[0]:.4f}")
    r = robot.left_arm.set_joint_positions(JointPositions(positions=target))
    print(f"    success={r.is_success}")
    print("    waiting 5s...")
    time.sleep(5)

    lj2 = robot.left_arm.get_joint_states()
    print(f"    joint0 now: {lj2.position[0]:.4f} (expected ~{target[0]:.4f})")
    moved = abs(lj2.position[0] - orig[0]) > 0.01
    print(f"    MOVED: {moved}")

    # Step 11: if still didn't move, try reset + recover
    if not moved:
        print("\n[11] Trying left_arm.reset()...")
        r = robot.left_arm.reset()
        print(f"    success={r.is_success}")
        time.sleep(3)
        lj3 = robot.left_arm.get_joint_states()
        print(f"    joint0 after reset: {lj3.position[0]:.4f}")

        print("\n[12] Trying set_joint_positions again...")
        r = robot.left_arm.set_joint_positions(JointPositions(positions=target))
        print(f"    success={r.is_success}")
        time.sleep(3)
        lj4 = robot.left_arm.get_joint_states()
        print(f"    joint0 now: {lj4.position[0]:.4f}")
        moved2 = abs(lj4.position[0] - orig[0]) > 0.01
        print(f"    MOVED: {moved2}")

    print("\nDone.")


if __name__ == "__main__":
    typer.run(main)
