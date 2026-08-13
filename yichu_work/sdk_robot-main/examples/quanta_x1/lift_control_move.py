"""量子1号Pro 升降机构控制示例。

演示如何控制量子1号Pro机器人的升降机构进行上下移动。
"""

import signal
from typing import Annotated
import typer
from x2robot import Robot, connect
from x2robot.sdk import LiftPosition, RobotModeParam, RobotWorkMode, ManipulatorControlModeParam, ManipulatorControlMode
import time


def move_to_absolute_position(robot: Robot, target: float):
    """移动到指定绝对位置。

    参数:
        robot: 已连接的 Robot 实例
        target: 目标位置（米）
    """
    cur_position = robot.lift.get_lift_position()
    print(f"当前升降位置: {cur_position.position:.4f}")

    robot.lift.set_lift_position(LiftPosition(position=target))
    time.sleep(0.5)

    cur_position = robot.lift.get_lift_position()
    print(f"移动后位置: {cur_position.position:.4f}")


def move_by_lift_position(robot: Robot, direction: str, distance: float):
    """通过相对位移控制升降机构。

    基于当前位置加上指定方向的偏移量作为目标位置。

    参数:
        robot: 已连接的 Robot 实例
        direction: 移动方向，"up"(上升) 或 "down"(下降)
        distance: 移动距离（米）
    """
    # 读取当前升降位置
    cur_position = robot.lift.get_lift_position()
    print(f"当前升降位置: {cur_position}")

    # 根据方向计算目标位置
    if direction == "up":
        lift_position = LiftPosition(position=cur_position.position + distance)
    elif direction == "down":
        lift_position = LiftPosition(position=cur_position.position - distance)
    else:
        print(f"未知方向: {direction}")
        return

    # 执行升降移动
    robot.lift.set_lift_position(lift_position)
    time.sleep(0.5)

    # 读取移动后的位置
    cur_position = robot.lift.get_lift_position()
    print(f"当前位置: {cur_position}")


def stream_lift_joint_states(robot: Robot):
    """实时流式读取升降机构关节状态并打印。

    参数:
        robot: 已连接的 Robot 实例
    """
    print("开始升降机构关节状态数据流...")
    print("按 Ctrl+C 停止")

    try:
        for joint_state in robot.lift.get_joint_states_stream():
            print(f"升降关节状态: {joint_state}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n数据流已停止。")


def main(
    server: Annotated[str, typer.Option(help="服务器地址，如 localhost:50051")] = "localhost:50051",
    action: Annotated[str, typer.Option(help="操作类型: move(运动), stream(数据流)")] = "move",
    direction: Annotated[str, typer.Option(help="移动方向: up(上升), down(下降)")] = "down",
    distance: Annotated[float, typer.Option(help="移动距离（米）")] = 0.05,
    target: Annotated[float | None, typer.Option(help="目标绝对位置（米），设置后忽略 direction/distance")] = None,
):
    """量子1号Pro 升降机构控制命令行工具。

    用法示例:
      python lift_control.py --action move --direction up --distance 0.1 --server 192.168.36.116:50051
      python lift_control.py --action move --direction down --distance 0.05
      python lift_control.py --action stream --server 192.168.36.116:50051

      0.31
    """
    robot = connect(f"x2://{server}")

    if action == "move":
        # 切换到 SDK 控制模式
        robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
        robot.robot_control.set_manipulator_control_mode(
            ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
        )
        if target is not None:
            move_to_absolute_position(robot, target)
        else:
            move_by_lift_position(robot, direction, distance)
    elif action == "stream":
        stream_lift_joint_states(robot)
    else:
        print(f"未知操作: {action}")
        print("有效操作: move, stream")
        return


if __name__ == "__main__":
    typer.run(main)
