from typing import Annotated
import typer
import time
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlModeParam, ManipulatorControlMode,
    LiftPosition,
)

def main(
    server: Annotated[str, typer.Option(help="server address")] = "localhost:50051",
):
    robot = connect(f"x2://{server}")

    try:
        robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    except:
        pass
    try:
        robot.robot_control.set_manipulator_control_mode(
            ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
        )
    except Exception as e:
        print(f"set_manipulator_control_mode: {e}")

    # 检查升降杆电机状态
    print("Checking lift motor status...")
    try:
        status = robot.lift.get_motor_status()
        print(f"  all_joints_healthy: {status.all_joints_healthy}")
        for i, m in enumerate(status.joint_motor_status):
            print(f"  motor[{i}]: state_code={m.state_code}, error_bit_code={m.error_bit_code}")
    except Exception as e:
        print(f"  get_motor_status failed: {e}")

    pos = robot.lift.get_lift_position()
    print(f"Current lift position: {pos.position:.3f}")

    # 先读一下手臂电机的状态做对比
    print("\nChecking left arm motor status...")
    try:
        status = robot.left_arm.get_motor_status()
        print(f"  all_joints_healthy: {status.all_joints_healthy}")
        for i, m in enumerate(status.joint_motor_status):
            print(f"  motor[{i}]: state_code={m.state_code}, error_bit_code={m.error_bit_code}")
    except Exception as e:
        print(f"  get_motor_status failed: {e}")

    print("\nDone.")

if __name__ == "__main__":
    typer.run(main)
