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

    pos = robot.lift.get_lift_position()
    print(f"Initial lift position: {pos.position:.3f}")

    # 尝试逐步下降，每次 0.01m
    for i in range(10):
        pos = robot.lift.get_lift_position()
        target = max(0.0, pos.position - 0.01)
        if target >= pos.position:
            print(f"Step {i+1}: already at bottom {pos.position:.3f}, stopping")
            break
        print(f"Step {i+1}: {pos.position:.3f} -> {target:.3f}")
        r = robot.lift.set_lift_position(LiftPosition(position=target))
        print(f"  result: {r.is_success}")
        time.sleep(1.5)
        pos = robot.lift.get_lift_position()
        print(f"  actual position: {pos.position:.3f}")
        if pos.position == target or abs(pos.position - target) < 0.002:
            print(f"  moved successfully")
        else:
            print(f"  did NOT move to target (stuck at {pos.position:.3f})")

    # 回到初始位置附近
    print("\nMoving back up to 0.10...")
    r = robot.lift.set_lift_position(LiftPosition(position=0.10))
    print(f"  result: {r.is_success}")
    time.sleep(2)
    pos = robot.lift.get_lift_position()
    print(f"  final position: {pos.position:.3f}")

if __name__ == "__main__":
    typer.run(main)
