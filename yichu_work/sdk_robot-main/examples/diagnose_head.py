from typing import Annotated
import typer
import time
from x2robot import connect
from x2robot.sdk import RobotModeParam, RobotWorkMode, JointPositions, HeadPose, LiftPosition

def main(
    server: Annotated[str, typer.Option(help="server address, e.g., localhost:50051")] = "localhost:50051",
):
    robot = connect(f"x2://{server}")

    try:
        robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    except Exception as e:
        print(f"set_work_mode: {e}")

    # 先读取头部当前姿态
    try:
        pose = robot.head.get_pose()
        print(f"Head pose BEFORE reset: pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}")
    except Exception as e:
        print(f"Failed to get head pose: {e}")

    # 把头部先偏转到一个明显的角度
    print("Moving head to yaw=0.5, pitch=0.2...")
    try:
        robot.head.set_pose(HeadPose(pitch=0.2, yaw=0.5))
        time.sleep(2)
        pose = robot.head.get_pose()
        print(f"Head pose AFTER moving: pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}")
    except Exception as e:
        print(f"Failed to move head: {e}")

    # 现在归零
    print("Resetting head to zero...")
    try:
        robot.head.set_pose(HeadPose(pitch=0.0, yaw=0.0))
        time.sleep(2)
        pose = robot.head.get_pose()
        print(f"Head pose AFTER reset: pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}")
    except Exception as e:
        print(f"Failed to reset head: {e}")

    # 检查是否有腰部
    print("Checking waist...")
    if hasattr(robot, 'waist'):
        print("  robot has waist attribute")
    else:
        print("  robot has NO waist attribute (X1 has no waist)")

    # 检查升降杆
    print("Checking lift...")
    if hasattr(robot, 'lift'):
        try:
            pos = robot.lift.get_lift_position()
            print(f"  lift position: {pos.position:.3f}")
            robot.lift.set_lift_position(LiftPosition(position=0.0))
            time.sleep(1)
            pos = robot.lift.get_lift_position()
            print(f"  lift position after set to 0: {pos.position:.3f}")
        except Exception as e:
            print(f"  lift error: {e}")
    else:
        print("  robot has NO lift attribute")

    print("Diagnostic done.")

if __name__ == "__main__":
    typer.run(main)
