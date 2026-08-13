from typing import Annotated
import typer
import time
from x2robot import connect, Robot
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlModeParam, ManipulatorControlMode,
    LiftPosition,
)

def main(
    server: Annotated[str, typer.Option(help="server address")] = "localhost:50051",
):
    robot = connect(f"x2://{server}")

    # 完全按照 lift_control.py 示例的初始化流程
    try:
        robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    except Exception as e:
        print(f"set_work_mode: {e}")
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )

    pos = robot.lift.get_lift_position()
    print(f"Initial lift position: {pos.position:.3f}")

    # 尝试向上移动到 0.20
    print("Moving lift UP to 0.20...")
    r = robot.lift.set_lift_position(LiftPosition(position=0.20))
    print(f"  set_lift_position is_success: {r.is_success}")
    time.sleep(3)
    pos = robot.lift.get_lift_position()
    print(f"  position after: {pos.position:.3f}")

    # 向下移动到 0.0
    print("Moving lift DOWN to 0.0...")
    r = robot.lift.set_lift_position(LiftPosition(position=0.0))
    print(f"  set_lift_position is_success: {r.is_success}")
    time.sleep(3)
    pos = robot.lift.get_lift_position()
    print(f"  position after: {pos.position:.3f}")

if __name__ == "__main__":
    typer.run(main)
