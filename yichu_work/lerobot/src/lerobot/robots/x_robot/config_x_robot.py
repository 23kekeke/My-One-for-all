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

    # Joint groups to enable
    enable_left_arm: bool = True
    enable_right_arm: bool = True
    enable_left_gripper: bool = True
    enable_right_gripper: bool = True
    enable_head: bool = True
    enable_lift: bool = True

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

    # ── Timestamp alignment ──
    # How many seconds of camera history to keep in the ring buffer for nearest-neighbor alignment
    max_camera_history: float = 1.0
    # Periodically log alignment offset (camera vs joint timestamp) for diagnostics
    log_alignment_stats: bool = False
