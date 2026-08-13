"""量子1号Pro 夹爪控制示例。

演示如何控制量子1号Pro机器人的夹爪开合。
夹爪位置范围: 0.0(闭合) ~ 0.6(张开)。
"""

from typing import Annotated
from x2robot import Robot
import typer
from x2robot import connect
from x2robot.sdk import GripperPosition
from time import sleep
from x2robot.sdk import RobotModeParam, RobotWorkMode
from x2robot.sdk import ManipulatorControlModeParam, ManipulatorControlMode


def move_gripper(robot: Robot, gripper: str):
    """演示夹爪的位置控制。

    依次将夹爪设置到 0.0(全闭)、0.3(半开)、0.6(全开) 三个位置。

    参数:
        robot: 已连接的 Robot 实例
        gripper: "left"(左夹爪) 或 "right"(右夹爪)
    """
    # 切换到 SDK 控制模式
    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )

    # 选择左/右夹爪控制器
    gripper_controller = robot.left_gripper if gripper == "left" else robot.right_gripper

    # 读取当前位置
    position = gripper_controller.get_position()
    print(f"当前夹爪位置: {position}")

    # 闭合夹爪
    print("设置夹爪位置为 0.0（全闭）")
    gripper_controller.set_position(GripperPosition(position=0.0))
    sleep(1.5)
    position = gripper_controller.get_position()
    print(f"当前夹爪位置: {position}")

    # 半开
    print("设置夹爪位置为 0.3（半开）")
    gripper_controller.set_position(GripperPosition(position=0.3))
    sleep(1.0)
    position = gripper_controller.get_position()
    print(f"当前夹爪位置: {position}")

    # 全开
    print("设置夹爪位置为 0.6（全开）")
    gripper_controller.set_position(GripperPosition(position=0.6))
    sleep(1.0)
    position = gripper_controller.get_position()
    print(f"当前夹爪位置: {position}")


def stream_gripper_data(robot: Robot, gripper: str):
    """实时流式读取夹爪关节状态并打印。

    参数:
        robot: 已连接的 Robot 实例
        gripper: "left"(左夹爪) 或 "right"(右夹爪)
    """
    print("开始夹爪数据流...")
    print("按 Ctrl+C 停止")

    gripper_controller = robot.left_gripper if gripper == "left" else robot.right_gripper

    try:
        for joint_state in gripper_controller.get_joint_states_stream():
            print(f"夹爪关节状态: {joint_state}")
            sleep(0.1)
    except KeyboardInterrupt:
        print("\n数据流已停止。")


def main(
    server: Annotated[str, typer.Option(help="服务器地址，如 localhost:50051")] = "localhost:50051",
    action: Annotated[str, typer.Option(help="操作类型: move(运动), stream(数据流)")] = "move",
    gripper: Annotated[str, typer.Option(help="夹爪选择: left(左), right(右)")] = "left",
):
    """量子1号Pro 夹爪控制命令行工具。

    用法示例:
      python gripper_control.py --action move --gripper right --server 192.168.36.116:50051
      python gripper_control.py --action stream --gripper right --server 192.168.36.116:50051
    """
    robot = connect(f"x2://{server}")

    if action == "move":
        move_gripper(robot, gripper)
    elif action == "stream":
        stream_gripper_data(robot, gripper)
    else:
        print(f"未知操作: {action}")
        print("有效操作: move, stream")
        return


if __name__ == "__main__":
    typer.run(main)
