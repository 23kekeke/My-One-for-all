import time, math
import typer
from typing import Annotated
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlMode, ManipulatorControlModeParam,
    JointPositions, GripperPosition, LiftPosition, HeadPose,
    ChassisControlMode, ChassisControlModeParam, ChassisVelocity
)


def check(step_name, result, obj=None):
    ok = result.is_success if hasattr(result, 'is_success') else True
    msg = result.error_message if hasattr(result, 'error_message') else ''
    extra = f" value={obj}" if obj is not None else ''
    status = "✓" if ok else "✗"
    print(f"    {status} {step_name}{extra}")
    if not ok and msg:
        print(f"       error: {msg}")
    return ok


def main(
    server: Annotated[str, typer.Option(help="Robot server address")] = "localhost:50051",
    head: Annotated[bool, typer.Option(help="头部回中")] = False,
    lift_to: Annotated[float, typer.Option(help="腰部移动到绝对位置（米），如 0.304")] = -1.0,
):
    print(f"\n[1] Connecting to robot {server}...")
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"    Model: {model}")

    print(f"\n[2] Robot status...")
    try:
        info = robot.system.get_dynamic_info()
        print(f"    Power: {info.power_status}%")
    except Exception as e:
        print(f"    get_dynamic_info failed: {e}")

    print(f"\n[3] set_work_mode(SDK)...")
    r = robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    check("set_work_mode(SDK)", r)

    print(f"\n[4] set_manipulator_control_mode(JOINT_POSITIONS)...")
    try:
        r = robot.robot_control.set_manipulator_control_mode(
            ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
        )
        check("set_manipulator_control_mode", r)
    except Exception as e:
        print(f"    ✗ Exception: {e}")

    print(f"\n[5] recover_emergency_stop...")
    try:
        r = robot.robot_control.recover_emergency_stop()
        check("recover_emergency_stop", r)
    except Exception as e:
        print(f"    recover_emergency_stop N/A: {e}")

    print(f"\n[6] homing (机械臂归零测试)...")
    yn = input("    继续 homing? (y/N): ")
    if yn.lower() == 'y':
        try:
            r = robot.robot_control.homing()
            check("homing", r)
            if r.is_success:
                print("    等待归零完成...")
                time.sleep(2)
        except Exception as e:
            print(f"    ✗ homing failed: {e}")
    else:
        print("    跳过")

    print(f"\n[7] 读取当前关节位置...")
    try:
        lj = robot.left_arm.get_joint_states()
        rj = robot.right_arm.get_joint_states()
        print(f"    左臂: {[f'{p:.4f}' for p in lj.position]}")
        print(f"    右臂: {[f'{p:.4f}' for p in rj.position]}")
    except Exception as e:
        print(f"    失败: {e}")

    print(f"\n[8] 左臂 joint0 +0.05rad 微小移动测试...")
    try:
        lj = robot.left_arm.get_joint_states()
        orig = list(lj.position)
        target = orig.copy()
        target[0] += 0.05
        r = robot.left_arm.set_joint_positions(JointPositions(positions=target))
        check("set_joint_positions", r)
        if r.is_success:
            print("    发送成功，等待 3s...")
            time.sleep(3)
            lj2 = robot.left_arm.get_joint_states()
            print(f"    joint0: {orig[0]:.4f} → {lj2.position[0]:.4f}  (期望 ~{target[0]:.4f})")
            r2 = robot.left_arm.set_joint_positions(JointPositions(positions=orig))
            check("恢复原位", r2)
    except Exception as e:
        print(f"    失败: {e}")

    if lift_to >= 0:
        print(f"\n[9] 腰部移动到 {lift_to:.3f}m...")
        tolerance = 0.005
        for i in range(30):
            try:
                pos = robot.lift.get_lift_position()
                error = abs(pos.position - lift_to)
                print(f"    attempt {i+1}: lift={pos.position:.4f}m, target={lift_to:.3f}m, error={error:.4f}m")
                if error <= tolerance:
                    print("    ✓ 到达目标位置")
                    break
                r = robot.lift.set_lift_position(LiftPosition(position=lift_to))
                if not r.is_success:
                    print(f"    ✗ set_lift_position failed: {r.error_message}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    ✗ lift move failed: {e}")
                time.sleep(1)

    if head:
        print(f"\n[10] 头部回中...")
        try:
            robot.head.set_pose(HeadPose(pitch=0.0, yaw=0.0))
            print("    ✓ head reset done")
        except Exception as e:
            print(f"    ✗ head reset failed: {e}")
        time.sleep(2)

    print(f"\n诊断完成。")


if __name__ == "__main__":
    typer.run(main)
