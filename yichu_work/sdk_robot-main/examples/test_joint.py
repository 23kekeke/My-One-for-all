import time, sys
from x2robot import connect
from x2robot.sdk import RobotModeParam, RobotWorkMode, ManipulatorControlModeParam, ManipulatorControlMode, JointPositions

server = sys.argv[1]
joint_idx = int(sys.argv[2])  # 0~5 -> J1~J6
angle = float(sys.argv[3])

robot = connect(f"x2://{server}")
robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
robot.robot_control.set_manipulator_control_mode(
    ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
)

print(f"Joint {joint_idx+1} -> {angle} rad")
robot.left_arm.set_joint_positions(JointPositions(positions=[0]*6))
time.sleep(1)

pos = [0]*6
pos[joint_idx] = angle
robot.left_arm.set_joint_positions(JointPositions(positions=pos))
time.sleep(2)

robot.left_arm.set_joint_positions(JointPositions(positions=[0]*6))
print("Done")
