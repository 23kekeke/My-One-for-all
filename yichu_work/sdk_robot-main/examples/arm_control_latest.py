"""从配置文件读取14维数组控制左右臂+夹爪。

用法:
  python arm_control_config.py                       # 默认归零
  python arm_control_config.py --action target        # 执行配置姿势
  python arm_control_config.py --action custom --array ...  # 自定义
   
 --action custom --array 通过 typer 的 list[float] 类型接收14个连续数值，直接在命令行按顺序输入即可：
  python arm_control_config.py \
  --action custom \
  --array \
  -0.1486 0.4707 -0.8101 0.635 0.3164 0 \
  -0.1486 0.4707 -0.8101 0.635 0.3164 0 \
  0.3 0.3
14维数组顺序: [左J1, 左J2, 左J3, 左J4, 左J5, 左J6, 右J1, 右J2, 右J3, 右J4, 右J5, 右J6, 左夹爪, 右夹爪]

"""

import json, time
from typing import Annotated, Optional
from pathlib import Path
import typer
from x2robot import Robot, connect
from x2robot.sdk import RobotModeParam, RobotWorkMode
from x2robot.sdk import ManipulatorControlModeParam, ManipulatorControlMode
from x2robot.sdk import JointPositions, GripperPosition

import numpy as np
import toppra as ta
import toppra.algorithm as algo
from toppra.constraint import JointVelocityConstraint, JointAccelerationConstraint


ARM_LOWER_LIMITS = np.array([-2.792,  0.0, -3.14, -1.57, -1.4, -1.745])
ARM_UPPER_LIMITS = np.array([ 2.792,  3.44,  0.0,   1.57,  1.4,  1.745])
GRIPPER_LIMITS = (0.0, 0.6)
TRAJECTORY_DT = 0.002

CONFIG_PATH = Path(__file__).parent / "arm_pose.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def move_arm_toppra(arm, target_positions: list, v_max=1.0, a_max=3):
    lower, upper = ARM_LOWER_LIMITS, ARM_UPPER_LIMITS
    current = arm.get_joint_states()
    q_start = np.array(current.position)
    q_end = np.array(target_positions)
    if np.any(q_end < lower) or np.any(q_end > upper):
        print(f"Error: target out of limits"); return
    q_start = np.clip(q_start, lower, upper)
    n = len(q_start)
    path = ta.SplineInterpolator([0, 1], np.stack([q_start, q_end]))
    instance = algo.TOPPRA(
        [JointVelocityConstraint([v_max]*n), JointAccelerationConstraint([a_max]*n)], path)
    traj = instance.compute_trajectory(0, 0)
    if traj is None:
        print("TOPP-RA failed."); return
    for t in np.arange(0, traj.duration, TRAJECTORY_DT):
        q = np.clip(traj(t), lower, upper)
        arm.set_joint_positions(JointPositions(positions=q.tolist()))
        time.sleep(TRAJECTORY_DT)
    arm.set_joint_positions(JointPositions(positions=q_end.tolist()))


def set_gripper(gripper, position: float):
    pos = np.clip(position, *GRIPPER_LIMITS)
    gripper.set_position(GripperPosition(position=float(pos)))


def execute(robot, left_joints, right_joints, left_grip, right_grip):
    print(f"Left arm:  {left_joints}")
    print(f"Right arm: {right_joints}")
    print(f"Grippers:  L={left_grip:.3f}  R={right_grip:.3f}")
    move_arm_toppra(robot.left_arm, left_joints);   time.sleep(0.5)
    move_arm_toppra(robot.right_arm, right_joints); time.sleep(0.5)
    set_gripper(robot.left_gripper, left_grip);     time.sleep(0.3)
    set_gripper(robot.right_gripper, right_grip)
    print("Done.")


def main(
    server: Annotated[str, typer.Option(help="server address")] = "localhost:50051",
    action: Annotated[str, typer.Option(help="default(归零) / target(配置文件) / custom(自定义)")] = "default",
    array: Annotated[Optional[list[float]], typer.Option(help="14 values: 6L+6R+lg+rg")] = None,
):
    robot = connect(f"x2://{server}")
    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )

    if action == "default":
        execute(robot, [0]*6, [0]*6, 0.0, 0.0)
    elif action == "target":
        cfg = load_config()
        arr = cfg["target"]
        execute(robot, arr[0:6], arr[6:12], arr[12], arr[13])
    elif action == "custom":
        if not array or len(array) != 14:
            print("Error: --array needs 14 values"); raise typer.Exit()
        execute(robot, array[0:6], array[6:12], array[12], array[13])
    else:
        print("Invalid action. Use: default, target, custom")


if __name__ == "__main__":
    typer.run(main)
