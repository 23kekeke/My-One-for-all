from dataclasses import dataclass, field

from ..config import RobotConfig


@dataclass
class XRobotCameraConfig:
    """Camera metadata for x_robot cameras (accessed via gRPC, not standard camera drivers)."""

    width: int = 640
    height: int = 480
    fps: int = 30


@RobotConfig.register_subclass("x_robot")
@dataclass
class XRobotConfig(RobotConfig):
    server: str = "192.168.36.116:50051"

    # Auto-detected from robot on connect if None: "quanta_x1", "quanta_x2", "desktop"
    robot_model: str | None = None

    # Joint groups to enable
    enable_left_arm: bool = True
    enable_right_arm: bool = True
    enable_left_gripper: bool = True
    enable_right_gripper: bool = True
    enable_head: bool = True
    # None = auto-detect based on robot model (quanta_x1 -> lift, quanta_x2 -> waist, desktop -> neither)
    enable_lift: bool | None = None
    enable_waist: bool | None = None

    # Camera enable flags
    enable_head_camera: bool = True
    enable_left_arm_camera: bool = True
    enable_right_arm_camera: bool = True

    # Camera configurations (for resolution / feature shape metadata)
    head_camera: XRobotCameraConfig = field(default_factory=XRobotCameraConfig)
    left_arm_camera: XRobotCameraConfig = field(default_factory=XRobotCameraConfig)
    right_arm_camera: XRobotCameraConfig = field(default_factory=XRobotCameraConfig)

    # Control mode: "joint" or "end_pose"
    ctrl_mode: str = "joint"

    # Maximum relative target magnitude for safety clamping (None = no limit)
    max_relative_target: float | None = None

    # Velocity and acceleration limits for streaming control (from SDK arm_control.py)
    max_joint_velocity: float = 1.0     # rad/s
    max_joint_acceleration: float = 3.0  # rad/s^2
