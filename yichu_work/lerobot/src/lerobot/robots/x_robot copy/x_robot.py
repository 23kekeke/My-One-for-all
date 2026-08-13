import logging
import threading

from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..robot import Robot
from .config_x_robot import XRobotConfig

logger = logging.getLogger(__name__)

# Default joint counts per robot model component
_DEFAULT_JOINT_COUNTS: dict[str, dict[str, int]] = {
    "quanta_x1": {
        "left_arm": 6,
        "right_arm": 6,
        "left_gripper": 1,
        "right_gripper": 1,
        "head": 2,
        "lift": 1,
    },
    "quanta_x2": {
        "left_arm": 6,
        "right_arm": 6,
        "left_gripper": 1,
        "right_gripper": 1,
        "head": 2,
        "waist": 4,
    },
    "desktop": {
        "left_arm": 6,
        "right_arm": 6,
        "left_gripper": 1,
        "right_gripper": 1,
    },
}

# Joint limits per robot model (radians), from SDK examples/quanta_x1/arm_control.py
_JOINT_LIMITS: dict[str, dict[str, np.ndarray]] = {
    "quanta_x1": {
        "arm_lower": np.array([-2.792, 0.0, -3.14, -1.57, -1.4, -1.745]),
        "arm_upper": np.array([2.792, 3.44, 0.0, 1.57, 1.4, 1.745]),
    },
    "quanta_x2": {
        "arm_lower": np.array([-2.792, 0.0, -3.14, -1.57, -1.4, -1.745]),
        "arm_upper": np.array([2.792, 3.44, 0.0, 1.57, 1.4, 1.745]),
    },
    "desktop": {
        "arm_lower": np.array([-2.792, 0.0, -3.14, -1.57, -1.4, -1.745]),
        "arm_upper": np.array([2.792, 3.44, 0.0, 1.57, 1.4, 1.745]),
    },
}

# Gripper position range, from SDK examples/quanta_x1/gripper_control.py
_GRIPPER_LOWER = 0.0
_GRIPPER_UPPER = 0.6


class XRobot(Robot):
    config_class = XRobotConfig
    name = "x_robot"

    def __init__(self, config: XRobotConfig):
        super().__init__(config)
        self.config = config
        self._sdk_robot = None
        self.is_robot_connected = False

        # Motors dict: {motor_name: current_value}
        self.motors: dict[str, float] = {}
        # Camera keys in order
        self._camera_keys: list[str] = []
        # Resolved model name (set after connect)
        self._detected_model: str | None = config.robot_model

        # Streaming cache (populated by background threads)
        self._stream_cache: dict[str, Any] = {}
        self._stream_lock = threading.Lock()
        self._stop_streams = threading.Event()
        self._first_data_ready = threading.Event()
        self._stream_threads: list[threading.Thread] = []

        self._build_motor_dict()
        self._build_camera_keys()

    def _build_motor_dict(self) -> None:
        """Build the motors dictionary based on config flags and model defaults."""
        model = self._detected_model or "quanta_x1"

        if self.config.enable_left_arm:
            n_joints = _DEFAULT_JOINT_COUNTS.get(model, {}).get("left_arm", 6)
            for i in range(n_joints):
                self.motors[f"left_arm_{i}.pos"] = 0.0

        if self.config.enable_right_arm:
            n_joints = _DEFAULT_JOINT_COUNTS.get(model, {}).get("right_arm", 6)
            for i in range(n_joints):
                self.motors[f"right_arm_{i}.pos"] = 0.0

        if self.config.enable_left_gripper:
            self.motors["left_gripper.pos"] = 0.0

        if self.config.enable_right_gripper:
            self.motors["right_gripper.pos"] = 0.0

        if self.config.enable_head:
            self.motors["head_0.pos"] = 0.0
            self.motors["head_1.pos"] = 0.0

        # Resolve lift/waist auto-detection
        enable_lift = self.config.enable_lift
        enable_waist = self.config.enable_waist
        if enable_lift is None and enable_waist is None:
            if model == "quanta_x1":
                enable_lift = True
                enable_waist = False
            elif model == "quanta_x2":
                enable_lift = False
                enable_waist = True
            else:
                enable_lift = False
                enable_waist = False

        if enable_lift:
            self.motors["lift.pos"] = 0.0
        if enable_waist:
            n_joints = _DEFAULT_JOINT_COUNTS.get(model, {}).get("waist", 4)
            for i in range(n_joints):
                self.motors[f"waist_{i}.pos"] = 0.0

    def _build_camera_keys(self) -> None:
        if self.config.enable_head_camera:
            self._camera_keys.append("head_camera")
        if self.config.enable_left_arm_camera:
            self._camera_keys.append("left_arm_camera")
        if self.config.enable_right_arm_camera:
            self._camera_keys.append("right_arm_camera")

    def _get_camera_shape(self, camera_key: str) -> tuple[int, int, int]:
        cfg_map = {
            "head_camera": self.config.head_camera,
            "left_arm_camera": self.config.left_arm_camera,
            "right_arm_camera": self.config.right_arm_camera,
        }
        cfg = cfg_map.get(camera_key)
        if cfg is not None:
            return (cfg.height, cfg.width, 3)
        return (480, 640, 3)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {k: float for k in self.motors.keys()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: self._get_camera_shape(cam) for cam in self._camera_keys}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.is_robot_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from x2robot import connect as x2_connect
        except ImportError:
            raise ImportError(
                "x2robot package is required for XRobot. Install it with: pip install x2robot"
            )

        self._sdk_robot = x2_connect(f"x2://{self.config.server}")

        # Detect model and reconfigure if auto-detect was used
        detected = self._sdk_robot.get_robot_model()
        logger.info(f"Connected to {detected} at {self.config.server}")

        if self.config.robot_model is None:
            self._detected_model = detected
            # Rebuild motors if model changed from default
            self.motors.clear()
            self._build_motor_dict()
            # Clear cached properties
            for attr in ("observation_features", "action_features"):
                self.__dict__.pop(attr, None)

        self.is_robot_connected = True

        # Set SDK control mode for action sending
        self.configure()

        logger.info(f"{self} connected.")

    def configure(self) -> None:
        if self._sdk_robot is None:
            return
        try:
            from x2robot.sdk import (
                ManipulatorControlMode,
                ManipulatorControlModeParam,
                RobotModeParam,
                RobotWorkMode,
            )

            self._sdk_robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
            if self.config.ctrl_mode == "end_pose":
                self._sdk_robot.robot_control.set_manipulator_control_mode(
                    ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_END_POSE)
                )
            else:
                self._sdk_robot.robot_control.set_manipulator_control_mode(
                    ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
                )
        except Exception as e:
            logger.warning(f"Failed to configure robot control mode: {e}")

    def _decode_image(self, compressed_image: Any) -> np.ndarray:
        """Decode a gRPC CompressedImage to an RGB numpy array (H, W, 3)."""
        import cv2

        data = bytes(compressed_image.data)
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode camera image")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── gRPC streaming workers ────────────────────────────────────────

    def _camera_stream_worker(self, cam_key: str) -> None:
        """Background thread: read camera stream and cache the latest frame."""
        try:
            if cam_key == "head_camera":
                stream = self._sdk_robot.head_camera.get_rgb_video_stream(timeout=None)
            elif cam_key == "left_arm_camera":
                stream = self._sdk_robot.left_arm_camera.get_video_stream(timeout=None)
            elif cam_key == "right_arm_camera":
                stream = self._sdk_robot.right_arm_camera.get_video_stream(timeout=None)
            else:
                logger.warning(f"Unknown camera key: {cam_key}")
                return

            for frame_msg in stream:
                if self._stop_streams.is_set():
                    break
                if not frame_msg or not frame_msg.data:
                    continue
                try:
                    img = self._decode_image(frame_msg)
                    with self._stream_lock:
                        self._stream_cache[cam_key] = img
                except Exception:
                    logger.debug(f"Failed to decode {cam_key} frame", exc_info=True)
        except Exception as e:
            logger.warning(f"{cam_key} stream stopped: {e}")

    def _joint_stream_worker(self, name: str, stream_func) -> None:
        """Background thread: read joint state stream and cache latest values."""
        try:
            stream = stream_func(timeout=None)
            for state_msg in stream:
                if self._stop_streams.is_set():
                    break
                try:
                    positions = list(state_msg.position) if state_msg.position else None
                    with self._stream_lock:
                        self._stream_cache[name] = positions
                except Exception:
                    logger.debug(f"Failed to parse {name} joint state", exc_info=True)
        except Exception as e:
            logger.warning(f"{name} stream stopped: {e}")

    def _build_joint_name_mapping(self, names: list[str]) -> dict[str, str]:
        """Build mapping from SDK joint names to internal motor keys.

        SDK naming conventions observed:
          - 'left_arm_joint1' .. 'left_arm_joint6'  →  'left_arm_0.pos' .. 'left_arm_5.pos'
          - 'right_arm_joint1' .. 'right_arm_joint6' →  'right_arm_0.pos' .. 'right_arm_5.pos'
          - 'left_arm_gripper'   →  'left_gripper.pos'
          - 'right_arm_gripper'  →  'right_gripper.pos'
          - 'lift_joint'         →  'lift.pos'
          - 'head_pitch_joint'   →  'head_0.pos'
          - 'head_yaw_joint'     →  'head_1.pos'
          - 'waist_joint1' .. 'waist_jointN'        →  'waist_0.pos' .. 'waist_{N-1}.pos'
        """
        import re

        mapping: dict[str, str] = {}

        # Hard-coded special cases for parts whose SDK names don't follow
        # the generic "{prefix}_joint_{N}" pattern.
        SPECIAL_NAMES: dict[str, str] = {
            "left_arm_gripper": "left_gripper.pos",
            "right_arm_gripper": "right_gripper.pos",
            "head_pitch_joint": "head_0.pos",
            "head_yaw_joint": "head_1.pos",
        }

        for name in names:
            if name in SPECIAL_NAMES:
                key = SPECIAL_NAMES[name]
                if key in self.motors:
                    mapping[name] = key
                continue

            # Pattern: {prefix}_joint{N}  →  {prefix}_{N-1}.pos
            # Pattern: {prefix}_joint      →  {prefix}.pos
            m = re.match(r"(.+)_joint(\d*)$", name)
            if m:
                prefix = m.group(1)
                idx_str = m.group(2)
                if idx_str:
                    idx = int(idx_str) - 1
                    key = f"{prefix}_{idx}.pos"
                else:
                    key = f"{prefix}.pos"
                if key in self.motors:
                    mapping[name] = key

        if not mapping:
            logger.warning(f"Could not map any joint names to motor keys. Names: {names}, Motors: {list(self.motors.keys())}")
        else:
            unmapped = [n for n in names if n not in mapping]
            if unmapped:
                logger.debug(
                    f"{len(unmapped)}/{len(names)} joint names could not be mapped: {unmapped}. "
                    f"Mapped: {mapping}"
                )
        return mapping

    def _all_joints_stream_worker(self) -> None:
        """Background thread: read combined joint state stream and cache by name."""
        try:
            stream = self._sdk_robot.state.get_all_joint_states_stream(timeout=None)
            name_mapping: dict[str, str] | None = None
            _first_log_done = False

            for state_msg in stream:
                if self._stop_streams.is_set():
                    break
                try:
                    names = list(state_msg.name) if state_msg.name else []
                    positions = list(state_msg.position) if state_msg.position else []

                    # Debug: log first received raw data from SDK
                    if not _first_log_done and names:
                        _first_log_done = True
                        logger.info(
                            f"Received first joint state from SDK: "
                            f"names={names}, positions={[round(p, 4) for p in positions]}"
                        )

                    if name_mapping is None and names:
                        name_mapping = self._build_joint_name_mapping(names)

                    if name_mapping:
                        for sdk_name, pos in zip(names, positions):
                            motor_key = name_mapping.get(sdk_name)
                            if motor_key:
                                with self._stream_lock:
                                    self._stream_cache[motor_key] = float(pos)
                        self._first_data_ready.set()
                except Exception:
                    logger.debug("Failed to parse joint state", exc_info=True)
        except Exception as e:
            logger.warning(f"Joint state stream stopped: {e}")

    def _start_streams(self) -> None:
        """Start all background streaming threads and wait for first joint data."""
        self._stop_streams.clear()
        self._first_data_ready.clear()

        with self._stream_lock:
            self._stream_cache.clear()

        # Single combined joint state stream
        t = threading.Thread(
            target=self._all_joints_stream_worker,
            daemon=True,
            name="stream-all-joints",
        )
        self._stream_threads.append(t)
        t.start()

        # Camera streams
        for cam_key in self._camera_keys:
            t = threading.Thread(
                target=self._camera_stream_worker,
                args=(cam_key,),
                daemon=True,
                name=f"stream-{cam_key}",
            )
            self._stream_threads.append(t)
            t.start()

        logger.info(f"Started {len(self._stream_threads)} streaming threads")

        # Wait for first joint state data to arrive (with timeout)
        if self.motors:
            if not self._first_data_ready.wait(timeout=5.0):
                logger.warning("Timed out waiting for first joint state data from gRPC streams")
            else:
                logger.debug("First joint state data received from gRPC streams")

    def start_streams(self) -> None:
        """Start background streaming threads for cameras and joints."""
        self._start_streams()

    def stop_streams(self) -> None:
        """Stop all background streaming threads."""
        self._stop_stream_threads()

    def _stop_stream_threads(self) -> None:
        """Signal all streaming threads to stop and wait for them."""
        self._stop_streams.set()
        for t in self._stream_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._stream_threads.clear()

    def _read_arm_joints(self, side: str) -> list[float] | None:
        """Read joint positions from left or right arm. Returns None on failure."""
        try:
            arm = self._sdk_robot.left_arm if side == "left" else self._sdk_robot.right_arm
            state = arm.get_joint_states()
            return list(state.position) if state and state.position else None
        except Exception:
            logger.debug(f"Failed to read {side} arm joint states", exc_info=True)
            return None

    def _read_gripper_position(self, side: str) -> float | None:
        """Read gripper position. Returns None on failure."""
        try:
            gripper = self._sdk_robot.left_gripper if side == "left" else self._sdk_robot.right_gripper
            pos = gripper.get_position()
            return float(pos.position) if pos is not None else None
        except Exception:
            logger.debug(f"Failed to read {side} gripper position", exc_info=True)
            return None

    def _read_head_pose(self) -> tuple[float, float] | None:
        """Read head pose (pitch, yaw). Returns None on failure."""
        try:
            pose = self._sdk_robot.head.get_pose()
            return (float(pose.pitch), float(pose.yaw))
        except Exception:
            logger.debug("Failed to read head pose", exc_info=True)
            return None

    def _read_lift_position(self) -> float | None:
        try:
            pos = self._sdk_robot.lift.get_lift_position()
            return float(pos.position) if pos is not None else None
        except Exception:
            logger.debug("Failed to read lift position", exc_info=True)
            return None

    def _read_waist_joints(self) -> list[float] | None:
        try:
            state = self._sdk_robot.waist.get_joint_states()
            return list(state.position) if state and state.position else None
        except Exception:
            logger.debug("Failed to read waist joint states", exc_info=True)
            return None

    def _read_camera_image(self, camera_key: str) -> np.ndarray | None:
        """Read a single camera image. Returns None on failure."""
        try:
            if camera_key == "head_camera":
                img = self._sdk_robot.head_camera.get_rgb_image()
            elif camera_key == "left_arm_camera":
                img = self._sdk_robot.left_arm_camera.get_raw_image()
            elif camera_key == "right_arm_camera":
                img = self._sdk_robot.right_arm_camera.get_raw_image()
            else:
                return None
            if img is None or not img.data:
                return None
            return self._decode_image(img)
        except Exception:
            logger.debug(f"Failed to read {camera_key}", exc_info=True)
            return None

    def _read_all_joints_blocking(self) -> dict[str, float]:
        """Non-streaming fallback: read all joint positions via individual SDK calls."""
        result: dict[str, float] = {}

        model = self._detected_model or "quanta_x1"

        if self.config.enable_left_arm:
            positions = self._read_arm_joints("left")
            if positions:
                n = _DEFAULT_JOINT_COUNTS.get(model, {}).get("left_arm", 6)
                for i in range(min(len(positions), n)):
                    result[f"left_arm_{i}.pos"] = float(positions[i])

        if self.config.enable_right_arm:
            positions = self._read_arm_joints("right")
            if positions:
                n = _DEFAULT_JOINT_COUNTS.get(model, {}).get("right_arm", 6)
                for i in range(min(len(positions), n)):
                    result[f"right_arm_{i}.pos"] = float(positions[i])

        if self.config.enable_left_gripper:
            pos = self._read_gripper_position("left")
            if pos is not None:
                result["left_gripper.pos"] = float(pos)

        if self.config.enable_right_gripper:
            pos = self._read_gripper_position("right")
            if pos is not None:
                result["right_gripper.pos"] = float(pos)

        if self.config.enable_head:
            pose = self._read_head_pose()
            if pose is not None:
                result["head_0.pos"] = float(pose[0])
                result["head_1.pos"] = float(pose[1])

        if "lift.pos" in self.motors:
            pos = self._read_lift_position()
            if pos is not None:
                result["lift.pos"] = float(pos)

        if any(k.startswith("waist_") for k in self.motors):
            positions = self._read_waist_joints()
            if positions:
                n = _DEFAULT_JOINT_COUNTS.get(model, {}).get("waist", 4)
                for i in range(min(len(positions), n)):
                    result[f"waist_{i}.pos"] = float(positions[i])

        return result

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._stream_lock:
            cache = self._stream_cache.copy()

        # Start with default values, then override from stream cache
        obs_dict: dict[str, Any] = dict(self.motors)
        for k in self.motors:
            if k in cache:
                obs_dict[k] = cache[k]

        # Fallback: if cache is empty for ALL motors, try blocking reads
        if all(obs_dict[k] == 0.0 for k in self.motors):
            fallback_read = self._read_all_joints_blocking()
            if fallback_read:
                for k, v in fallback_read.items():
                    obs_dict[k] = v
                    with self._stream_lock:
                        self._stream_cache[k] = v

        # Populate camera images from stream cache
        for cam_key in self._camera_keys:
            img = cache.get(cam_key)
            if img is not None:
                obs_dict[cam_key] = img
            elif cam_key not in obs_dict:
                # Return a black frame if stream hasn't produced a frame yet
                obs_dict[cam_key] = np.zeros(self._get_camera_shape(cam_key), dtype=np.uint8)

        # Update internal motor state
        for k in self.motors:
            if k in obs_dict:
                self.motors[k] = obs_dict[k]

        return obs_dict


    def _clip_arm_joints(self, positions: np.ndarray, side: str) -> np.ndarray:
        """Clip arm joint positions to hardware limits, following SDK arm_control.py.

        Logs a warning when any joint value is out of bounds.
        """
        model = self._detected_model or "quanta_x1"
        limits = _JOINT_LIMITS.get(model, _JOINT_LIMITS.get("quanta_x1", {}))
        lower = limits.get("arm_lower")
        upper = limits.get("arm_upper")
        if lower is None or upper is None:
            return positions

        clipped = np.clip(positions, lower[: len(positions)], upper[: len(positions)])
        for i in range(len(positions)):
            if not np.isclose(positions[i], clipped[i]):
                logger.warning(
                    f"{side}_arm_{i}.pos clipped from {positions[i]:.4f} to "
                    f"[{lower[i]:.4f}, {upper[i]:.4f}]"
                )
        return clipped

    def _clip_gripper(self, value: float, side: str) -> float:
        """Clip gripper position to [_GRIPPER_LOWER, _GRIPPER_UPPER], following SDK gripper_control.py."""
        clipped = float(np.clip(value, _GRIPPER_LOWER, _GRIPPER_UPPER))
        if not np.isclose(value, clipped):
            logger.warning(
                f"{side}_gripper.pos clipped from {value:.4f} to [{_GRIPPER_LOWER}, {_GRIPPER_UPPER}]"
            )
        return clipped

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """直接向机器人发送动作指令（无插值/限速，仅控制右臂和右夹爪）。

        控制规则：
        - 只对 ``action`` 中显式出现的组件 key 发送控制指令；
        - 对机械臂，传入部分关节时，未出现的关节从 motors 取当前值；
        - action 中完全没有该组件的 key 时，跳过该组件。

        示例：
            send_action({"right_gripper.pos": 0.3})
            send_action({"right_arm_0.pos": 0.5, "right_gripper.pos": 0.3})
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            from x2robot.sdk import GripperPosition, JointPositions
        except ImportError:
            raise ImportError("x2robot package is required for sending actions.")

        model = self._detected_model or "quanta_x1"
        sent_action: dict[str, float] = {}

        # ── 右臂 ──
        if self.config.enable_right_arm:
            n_joints = _DEFAULT_JOINT_COUNTS.get(model, {}).get("right_arm", 6)
            arm_keys = [f"right_arm_{i}.pos" for i in range(n_joints)]
            if any(k in action for k in arm_keys):
                positions = []
                for key in arm_keys:
                    if key in action:
                        positions.append(float(action[key]))
                    else:
                        positions.append(float(self.motors.get(key, 0.0)))
                positions_arr = np.array(positions)
                clipped = self._clip_arm_joints(positions_arr, "right")
                self._sdk_robot.right_arm.set_joint_positions(JointPositions(positions=clipped.tolist()))
                for i in range(n_joints):
                    sent_action[arm_keys[i]] = float(clipped[i])

        # ── 右夹爪 ──
        if "right_gripper.pos" in action and "right_gripper.pos" in self.motors:
            target = float(action["right_gripper.pos"])
            pos = self._clip_gripper(target, "right")
            self._sdk_robot.right_gripper.set_position(GripperPosition(position=pos))
            sent_action["right_gripper.pos"] = pos

        self.motors.update(sent_action)
        return self.motors.copy()

    def get_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return the latest joint positions as the action (for recording).

        Reads from stream cache first, falling back to internal motor state.
        The `action` parameter is merged in, allowing external inputs
        (e.g. teleop or policy) to override specific joints.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._stream_lock:
            cache = self._stream_cache.copy()

        result = dict(self.motors)
        for k in self.motors:
            if k in cache:
                result[k] = cache[k]
        # Merge any externally provided action values
        result.update({k: float(v) for k, v in action.items() if k in self.motors})
        return result

    @property
    def cameras(self) -> dict:
        """Compatibility property returning camera info for the record script."""
        result: dict[str, Any] = {}
        for key in self._camera_keys:
            result[key] = type("_CameraInfo", (), {
                "height": self._get_camera_shape(key)[0],
                "width": self._get_camera_shape(key)[1],
            })()
        return result

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.is_robot_connected = False
        self._stop_stream_threads()
        self._sdk_robot = None

        logger.info(f"{self} disconnected.")
