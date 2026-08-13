from typing import Annotated, Optional
import typer
import time
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlModeParam, ManipulatorControlMode,
    HeadPose, LiftPosition,
    ChassisControlMode, ChassisControlModeParam, ChassisPosition, ChassisVelocity,
)

HEAD_LIMITS = {
    "quanta_x1": {"pitch_min": -0.06, "pitch_max": 0.9, "yaw_min": -1.20, "yaw_max": 1.20},
    "quanta_x2": {"pitch_min": -0.52, "pitch_max": 0.87, "yaw_min": -1.57, "yaw_max": 1.57},
}

def clamp_head_value(model: str, value: float, axis: str) -> float:
    limits = HEAD_LIMITS.get(model)
    if limits is None:
        return value
    vmin, vmax = limits[f"{axis}_min"], limits[f"{axis}_max"]
    if value < vmin or value > vmax:
        clamped = max(vmin, min(vmax, value))
        print(f"  warning: {axis}={value:.2f}° out of range [{vmin:.2f}, {vmax:.2f}] for {model}, clamped to {clamped:.2f}°")
        return clamped
    return value

def main(
    server: Annotated[str, typer.Option(help="server address, e.g., localhost:50051")] = "localhost:50051",
    arms: Annotated[bool, typer.Option(help="双臂归零")] = False,
    head: Annotated[bool, typer.Option(help="头部回中")] = False,
    lift_zero: Annotated[bool, typer.Option(help="腰部归零（降到最低）")] = False,
    lift_move: Annotated[float, typer.Option(help="腰部相对移动（米），正=上升，负=下降，0=不动")] = 0.0,
    lift_to: Annotated[float, typer.Option(help="腰部移动到绝对位置（米），如 0.304")] = -1.0,
    head_pitch_to: Annotated[Optional[float], typer.Option(help="头部俯仰绝对位置（度），负值抬头，正值低头，如 -0.52 为最高")] = None,
    head_yaw_to: Annotated[Optional[float], typer.Option(help="头部偏航绝对位置（度），正值右转，负值左转，如 1.57 为最右")] = None,
    chassis_x: Annotated[float, typer.Option(help="底盘前后移动（米），正=前进，负=后退，如 0.1")] = 0.0,
    chassis_y: Annotated[float, typer.Option(help="底盘左右移动（米），正=右移，负=左移，如 0.1")] = 0.0,
    chassis_yaw: Annotated[float, typer.Option(help="底盘旋转（弧度），正=顺时针，负=逆时针，如 0.5")] = 0.0,
):
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"  detected model: {model}")

    for attempt in range(3):
        try:
            r = robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK), timeout=10)
            if r.is_success:
                break
            print(f"set_work_mode attempt {attempt + 1}/3: is_success={r.is_success}, error={r.error_message}")
        except Exception as e:
            print(f"set_work_mode attempt {attempt + 1}/3 failed: {e}")
            time.sleep(1)

    for attempt in range(3):
        try:
            r = robot.robot_control.set_manipulator_control_mode(
                ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS),
                timeout=10,
            )
            if r.is_success:
                break
            print(f"set_manipulator_control_mode attempt {attempt + 1}/3: is_success={r.is_success}")
        except Exception as e:
            print(f"set_manipulator_control_mode attempt {attempt + 1}/3 failed: {e}")
            time.sleep(1)

    if lift_zero:
        print("Resetting waist/lift to zero...")
        try:
            robot.robot_control.set_manipulator_control_mode(
                ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
            )
            pos = robot.lift.get_lift_position()
            print(f"  lift position before: {pos.position:.3f}")
            r = robot.lift.set_lift_position(LiftPosition(position=0.0))
            print(f"  set_lift_position(0.0) is_success: {r.is_success}")
            for _ in range(8):
                time.sleep(1)
                pos = robot.lift.get_lift_position()
                if pos.position < 0.01:
                    break
            print(f"  lift position after:  {pos.position:.3f}")
            print("  waist/lift reset done")
        except Exception as e:
            print(f"  waist/lift reset failed: {e}")
    elif lift_move != 0.0:
        print(f"Moving waist/lift by {lift_move:+.3f}m ...")
        try:
            robot.robot_control.set_manipulator_control_mode(
                ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
            )
            pos = robot.lift.get_lift_position()
            target = pos.position + lift_move
            print(f"  lift position before: {pos.position:.3f}")
            print(f"  target position: {target:.3f}")
            r = robot.lift.set_lift_position(LiftPosition(position=target))
            print(f"  set_lift_position is_success: {r.is_success}")
            time.sleep(3)
            pos = robot.lift.get_lift_position()
            print(f"  lift position after:  {pos.position:.3f}")
        except Exception as e:
            print(f"  waist/lift move failed: {e}")

    if lift_to >= 0:
        tolerance = 0.005
        max_attempts = 30
        for i in range(max_attempts):
            try:
                robot.robot_control.set_manipulator_control_mode(
                    ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
                )
                pos = robot.lift.get_lift_position()
                error = abs(pos.position - lift_to)
                print(f"Attempt {i+1}: lift={pos.position:.4f}m, target={lift_to:.3f}m, error={error:.4f}m")
                if error <= tolerance:
                    print(f"  Reached target position.")
                    break
                r = robot.lift.set_lift_position(LiftPosition(position=lift_to))
                if not r.is_success:
                    print(f"  set_lift_position failed: {r.error_message}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  waist/lift move failed (attempt {i+1}/{max_attempts}): {e}")
                time.sleep(1)
        else:
            print(f"  Failed to reach target after {max_attempts} attempts")

    if arms:
        print("Calling robot homing (arms)...")
        try:
            result = robot.robot_control.homing()
            print(f"  homing result: {result.is_success}")
        except Exception as e:
            print(f"  arms homing failed: {e}")
        time.sleep(2)

    if head_pitch_to is not None or head_yaw_to is not None:
        try:
            cur = robot.head.get_pose()
        except Exception:
            cur = HeadPose(pitch=0.0, yaw=0.0)
        pitch = head_pitch_to if head_pitch_to is not None else cur.pitch
        yaw = head_yaw_to if head_yaw_to is not None else cur.yaw
        pitch = clamp_head_value(model, pitch, "pitch")
        yaw = clamp_head_value(model, yaw, "yaw")
        print(f"Moving head to pitch={pitch:.2f}°, yaw={yaw:.2f}°...")
        try:
            robot.head.set_pose(HeadPose(pitch=pitch, yaw=yaw))
            print("  head move done")
        except Exception as e:
            print(f"  head move failed: {e}")
        time.sleep(2)

    if head:
        print("Resetting head to center...")
        try:
            robot.head.set_pose(HeadPose(pitch=0.0, yaw=0.0))
            print("  head reset done")
        except Exception as e:
            print(f"  head reset failed: {e}")
        time.sleep(2)

    if chassis_x != 0.0 or chassis_y != 0.0 or chassis_yaw != 0.0:
        if chassis_y != 0.0:
            print(f"  warning: --chassis-y={chassis_y:+.3f} 不支持（该机器人无横向移动能力），已忽略")

        if chassis_x != 0.0 or chassis_yaw != 0.0:
            robot.chassis.set_control_mode(ChassisControlModeParam(mode=ChassisControlMode.VELOCITY))
            if chassis_x != 0.0:
                speed = 0.1
                duration = abs(chassis_x) / speed
                direction = 1.0 if chassis_x > 0 else -1.0
                print(f"Moving chassis {chassis_x:+.3f}m at {speed}m/s for {duration:.1f}s...")
                try:
                    start = time.time()
                    while time.time() - start < duration:
                        robot.chassis.set_velocity(ChassisVelocity(vel_x=direction * speed, vel_y=0.0, vel_yaw=0.0))
                        time.sleep(0.05)
                    for _ in range(10):
                        robot.chassis.set_velocity(ChassisVelocity(vel_x=0.0, vel_y=0.0, vel_yaw=0.0))
                        time.sleep(0.05)
                except Exception as e:
                    print(f"  chassis move failed: {e}")
            if chassis_yaw != 0.0:
                speed = 0.3
                duration = abs(chassis_yaw) / speed
                direction = 1.0 if chassis_yaw > 0 else -1.0
                print(f"Rotating chassis {chassis_yaw:+.3f}rad at {speed}rad/s for {duration:.1f}s...")
                try:
                    start = time.time()
                    while time.time() - start < duration:
                        robot.chassis.set_velocity(ChassisVelocity(vel_x=0.0, vel_y=0.0, vel_yaw=direction * speed))
                        time.sleep(0.05)
                    for _ in range(10):
                        robot.chassis.set_velocity(ChassisVelocity(vel_x=0.0, vel_y=0.0, vel_yaw=0.0))
                        time.sleep(0.05)
                except Exception as e:
                    print(f"  chassis rotate failed: {e}")

    if not any([arms, head, lift_zero, lift_move != 0.0, lift_to >= 0, head_pitch_to is not None, head_yaw_to is not None, chassis_x != 0.0, chassis_y != 0.0, chassis_yaw != 0.0]):
        print("No operation specified.")

if __name__ == "__main__":
    typer.run(main)
