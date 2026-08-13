"""量子1号Pro 机械臂控制示例。

演示如何控制量子1号Pro机器人的6自由度机械臂，支持两种控制模式：
  - joint_pos: 关节角度控制，使用 TOPP-RA 进行轨迹规划
  - end_pose:  末端位姿控制，使用 SLERP 进行姿态插值
"""

import time
from typing import Annotated
import typer
from x2robot import Robot, connect
from x2robot.sdk import RobotModeParam, RobotWorkMode
from x2robot.sdk import ManipulatorControlModeParam, ManipulatorControlMode
from x2robot.geometry_msgs import Pose, Point, Quaternion
from x2robot.sdk import JointPositions

import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from toppra.constraint import JointVelocityConstraint, JointAccelerationConstraint


# 量子1号Pro 各关节限位（弧度）
ARM_LOWER_LIMITS = np.array([-2.792,  0.0, -3.14, -1.57, -1.4, -1.745])
ARM_UPPER_LIMITS = np.array([ 2.792,  3.44,  0.0,   1.57,  1.4,  1.745])

# TOPP-RA 轨迹执行的控制周期（500Hz）
TRAJECTORY_DT = 0.002


def move_arm_joints_toppra(arm, target_positions: list, v_max=1.0, a_max=3):
    """使用 TOPP-RA 算法将机械臂移动到目标关节角度。

    在速度和加速度约束下规划时间最优轨迹，以 500Hz 频率执行。

    参数:
        arm: 机械臂控制器对象（如 robot.left_arm）
        target_positions: 目标关节角度列表（弧度）
        v_max: 最大关节速度（rad/s），默认 1.0
        a_max: 最大关节加速度（rad/s^2），默认 3.0
    """
    lower_limits = ARM_LOWER_LIMITS
    upper_limits = ARM_UPPER_LIMITS

    # 1. 读取当前关节状态作为轨迹起点
    current_state = arm.get_joint_states()
    q_start = np.array(current_state.position)
    q_end = np.array(target_positions)

    # 校验目标位置是否在限位范围内
    if np.any(q_end < lower_limits) or np.any(q_end > upper_limits):
        print("Error: Target position out of limits!")
        return
    # 裁剪当前位置，防止传感器漂移导致起点越界
    q_start = np.clip(q_start, lower_limits, upper_limits)

    num_joints = len(q_start)

    # 2. 在关节空间中构建直线路径
    waypoints = np.stack([q_start, q_end])
    path = ta.SplineInterpolator([0, 1], waypoints)

    # 3. 定义动力学约束（速度和加速度限幅）
    pc_vel = JointVelocityConstraint([v_max] * num_joints)
    pc_acc = JointAccelerationConstraint([a_max] * num_joints)

    # 4. 求解时间参数化轨迹
    # 位置由路径保证，速度和加速度由约束限制
    instance = algo.TOPPRA([pc_vel, pc_acc], path)
    traj = instance.compute_trajectory(0, 0)

    if traj is None:
        print("TOPP-RA 轨迹规划失败。")
        return

    # 5. 以 500Hz 频率执行轨迹
    duration = traj.duration
    dt = TRAJECTORY_DT
    ts = np.arange(0, duration, dt)

    for t in ts:
        q_t = traj(t)
        # 二次保险：在指令层面裁剪数值
        q_t_safe = np.clip(q_t, lower_limits, upper_limits)

        joint_cmd = JointPositions()
        joint_cmd.positions = q_t_safe.tolist()
        arm.set_joint_positions(joint_cmd)
        time.sleep(dt)

    # 最后发送精确目标点，确保末端对准
    final_cmd = JointPositions()
    final_cmd.positions = q_end.tolist()
    arm.set_joint_positions(final_cmd)
    print("运动完成。")


def move_by_joint_positions(robot: Robot, arm):
    """演示通过关节角度模式控制机械臂。

    步骤:
    1. 将机械臂复位到零位
    2. 移动到预设的目标关节角度

    参数:
        robot: 已连接的 Robot 实例
        arm: "left" 或 "right"
    """
    # 切换到 SDK 控制模式，并设置为关节角度控制模式
    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )

    # --- 任务 A: 所有关节复位到零位 ---
    print("开始关节复位...")
    zero_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if arm == "left":
        move_arm_joints_toppra(robot.left_arm, zero_positions)
    elif arm == "right":
        move_arm_joints_toppra(robot.right_arm, zero_positions)
    else:
        print("无效的机械臂参数")
        return

    time.sleep(1)

    # --- 任务 B: 移动到特定关节角度（弧度） ---
    target_q = [-0.1486, 0.4707, -0.8101, 0.6350, 0.3164, 0.0]

    print(f"移动到目标关节角度: {target_q}")
    if arm == "left":
        move_arm_joints_toppra(robot.left_arm, target_q)
    elif arm == "right":
        move_arm_joints_toppra(robot.right_arm, target_q)
    else:
        print("无效的机械臂参数")
        return

    time.sleep(2)
    print("演示结束。")



def move_arm_endpose_toppra(arm, target_pose, v_max=2.2, a_max=0.3):
    """使用 TOPP-RA 算法将机械臂末端移动到目标位姿。

    对直线路径进行时间参数化，同时用 SLERP 对姿态进行球面线性插值。

    参数:
        arm: 机械臂控制器对象
        target_pose: 目标末端 Pose（位置 + 四元数姿态）
        v_max: 最大末端线速度（m/s），默认 2.2
        a_max: 最大末端线加速度（m/s^2），默认 0.3
    """
    # 1. 获取当前末端位姿作为绝对起点
    start_pose = arm.get_end_pose()
    
    p_start = np.array([start_pose.pose.position.x, start_pose.pose.position.y, start_pose.pose.position.z])
    p_end = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])
    
    q_start = [start_pose.pose.orientation.x, start_pose.pose.orientation.y, 
               start_pose.pose.orientation.z, start_pose.pose.orientation.w]
    q_end = [target_pose.orientation.x, target_pose.orientation.y, 
             target_pose.orientation.z, target_pose.orientation.w]

    # 2. TOPP-RA 轨迹规划
    dist = np.linalg.norm(p_end - p_start)
    path_len = dist if dist > 1e-6 else 1.0
    
    # 建立一维几何路径：从 0 移动到 path_len
    path = ta.SplineInterpolator([0, 1], np.array([[0], [path_len]]))
    pc_vel = constraint.JointVelocityConstraint([v_max])
    pc_acc = constraint.JointAccelerationConstraint([a_max])
    
    instance = algo.TOPPRA([pc_vel, pc_acc], path)
    traj = instance.compute_trajectory(0, 0)
    
    if traj is None:
        print("TOPP-RA 轨迹规划失败")
        return

    # 3. 准备姿态插值器（SLERP）
    key_rots = R.from_quat([q_start, q_end])
    slerp_func = Slerp([0, traj.duration], key_rots)

    # 4. 执行运动（200Hz）
    duration = traj.duration
    interval = 0.005
    ts = np.arange(0, duration, interval)
    
    print(f"执行 TOPP-RA 平滑轨迹: 预估耗时 {duration:.2f}s")

    for t in ts:
        # 获取当前时刻的规划位置 s_t（0 <= s_t <= path_len）
        s_t = traj(t)[0]
        
        # 计算非线性 alpha 比例系数（该系数满足平滑加减速）
        # 虽然路径是直线，但运动速度由 TOPP-RA 控制
        alpha = np.clip(s_t / path_len, 0, 1)

        pose = Pose()
        pose.position = Point()      
        pose.orientation = Quaternion() 
        
        # --- 位置插值：alpha 遵循速度曲线 ---
        pose.position.x = p_start[0] + (p_end[0] - p_start[0]) * alpha
        pose.position.y = p_start[1] + (p_end[1] - p_start[1]) * alpha
        pose.position.z = p_start[2] + (p_end[2] - p_start[2]) * alpha

        # --- 姿态插值（SLERP） ---
        curr_q = slerp_func(t).as_quat()
        pose.orientation.x = curr_q[0]
        pose.orientation.y = curr_q[1]
        pose.orientation.z = curr_q[2]
        pose.orientation.w = curr_q[3]

        # 发送末端位姿指令
        arm.set_end_pose(pose)
        time.sleep(interval)

    # 5. 确保末端精确到达目标点
    arm.set_end_pose(target_pose)
    time.sleep(0.1)

    print("运动完成")


def move_by_end_pose(robot: Robot, arm):
    """演示通过末端位姿模式控制机械臂。

    步骤:
    1. 将末端复位到零位姿
    2. 将末端向上移动 0.2 米

    参数:
        robot: 已连接的 Robot 实例
        arm: "left" 或 "right"
    """
    # 切换到 SDK 控制模式，并设置为末端位姿控制模式
    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_END_POSE)
    )

    # --- 任务 A: 末端位姿复位 ---
    print("开始末端位姿复位...")
    zero_pose = Pose()
    zero_pose.position = Point(x=0.0, y=0.0, z=0.0)
    zero_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    if arm == "left":
        move_arm_endpose_toppra(robot.left_arm, zero_pose)
    elif arm == "right":
        move_arm_endpose_toppra(robot.right_arm, zero_pose)
    else:
        print("无效的机械臂参数")
        return
    time.sleep(2)

    # --- 任务 B: 移动到目标末端位姿（向上 0.2 米） ---
    target = Pose()
    target.position = Point(x=0.0, y=0.0, z=0.2)
    target.orientation = Quaternion(x=-0.0076, y=0.0868, z=0.0868, w=0.9924)
    
    print("执行 TOPP-RA 轨迹控制...")
    if arm == "left":
        move_arm_endpose_toppra(robot.left_arm, target)
    elif arm == "right":
        move_arm_endpose_toppra(robot.right_arm, target)


def stream_arm_joint_states(robot: Robot, arm):
    """实时流式读取机械臂关节状态并打印。

    参数:
        robot: 已连接的 Robot 实例
        arm: "left" 或 "right"
    """
    print("开始机械臂关节状态数据流...")
    print("按 Ctrl+C 停止")
    try:
        if arm == "left":
            for joint_state in robot.left_arm.get_joint_states_stream():
                print(f"关节状态: {joint_state}")
                time.sleep(0.1)
        elif arm == "right":
            for joint_state in robot.right_arm.get_joint_states_stream():
                print(f"关节状态: {joint_state}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n数据流已停止。")


def stream_arm_end_pose(robot: Robot, arm):
    """实时流式读取机械臂末端位姿并打印。

    参数:
        robot: 已连接的 Robot 实例
        arm: "left" 或 "right"
    """
    print("开始机械臂末端位姿数据流...")
    print("按 Ctrl+C 停止")
    try:
        if arm == "left":
            for end_pose in robot.left_arm.get_end_pose_stream():
                print(f"末端位姿: {end_pose}")
                time.sleep(0.1)
        elif arm == "right":
            for end_pose in robot.right_arm.get_end_pose_stream():
                print(f"末端位姿: {end_pose}")
                time.sleep(0.1)
        else:
            print("无效的机械臂参数，可选值: left, right")
            return
    except KeyboardInterrupt:
        print("\n数据流已停止。")

def stream_arm_end_pose(robot: Robot, arm):
    print("Starting arm end pose streaming...")
    print("Press Ctrl+C to stop streaming")
    try:
        if arm == "left":
            for end_pose in robot.left_arm.get_end_pose_stream():
                print(f"end_pose: {end_pose}")
                time.sleep(0.1)
        elif arm == "right":
            for end_pose in robot.right_arm.get_end_pose_stream():
                print(f"end_pose: {end_pose}")
                time.sleep(0.1)
        else:
            print("Invalid arm. Valid options: left, right")
            return
    except KeyboardInterrupt:
        print("\nStopping streaming...")

def main(
    server: Annotated[str, typer.Option(help="服务器地址")] = "localhost:50051",
    action: Annotated[str, typer.Option(help="操作类型: move(运动), stream(数据流)")] = "move",
    mode: Annotated[str, typer.Option(help="控制模式: joint_pos(关节角度), end_pose(末端位姿)")] = "joint_pos",
    arm: Annotated[str, typer.Option(help="机械臂选择: left(左臂), right(右臂)")] = "left",
):
    """量子1号Pro 机械臂控制命令行工具。

    用法示例:
      python arm_control.py --action move --mode joint_pos --arm left --server 192.168.10.1:50051
      python arm_control.py --action stream --mode joint_pos --arm left --server 192.168.36.116:50051
    """
    print("该示例将把机械臂向上抬起0.2米，请确保机械臂路径无障碍物")
    if not input("继续? (y/n): ").lower() == "y":
        return

    # 连接机器人
    robot = connect(f"x2://{server}")

    if action == "move":
        if mode == "joint_pos":
            move_by_joint_positions(robot, arm)
        elif mode == "end_pose":
            move_by_end_pose(robot, arm)
        else:
            print("无效的模式，可选值: joint_pos, end_pose")
            return
    elif action == "stream":
        if mode == "joint_pos":
            stream_arm_joint_states(robot, arm)
        elif mode == "end_pose":
            stream_arm_end_pose(robot, arm)
        else:
            print("无效的模式，可选值: joint_pos, end_pose")
            return
    else:
        print("无效的操作，可选值: move, stream")
        return


if __name__ == "__main__":
    typer.run(main)