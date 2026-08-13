import logging
import threading
import time
from collections import deque
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..robot import Robot
from .config_x_robot import XRobotConfig

logger = logging.getLogger(__name__)

# 各机器人型号各组件的默认关节数量
# Default joint counts per robot model component
_DEFAULT_JOINT_COUNTS: dict[str, dict[str, int]] = {
    "quanta_x1": {
        "left_arm": 6,          # 左臂 6 关节
        "right_arm": 6,         # 右臂 6 关节
        "left_gripper": 1,      # 左夹爪 1 关节
        "right_gripper": 1,     # 右夹爪 1 关节
        # "head": 2,              # 头部 2 关节（pitch/yaw）
        # "lift": 1,              # 升降 1 关节
    },
}

# 每个关节的物理限位（rad），来自 x2robot SDK examples/arm_control_latest.py
# Joint position limits per joint (rad)
_ARM_LOWER_LIMITS = np.array([-2.792,  0.0, -3.14, -1.57, -1.4, -1.745])
_ARM_UPPER_LIMITS = np.array([ 2.792,  3.44,  0.0,   1.57,  1.4,  1.745])
_GRIPPER_LIMITS = (0.0, 1.5)


class XRobot(Robot):
    """XRobot 机器人控制类（仅支持 quanta_x1）。

    通过 gRPC 与机器人 SDK 通信，负责：
    - 连接/断开机器人
    - 流式读取关节状态和相机图像
    - 时间戳对齐（关节 500Hz 为基准，相机最近邻匹配）
    - 发送动作指令（双臂、夹爪、头部、升降）
    """
    config_class = XRobotConfig
    name = "x_robot"

    def __init__(self, config: XRobotConfig):
        super().__init__(config)
        self.config = config
        self._sdk_robot = None              # gRPC 机器人 SDK 对象
        self.is_robot_connected = False

        # 电机字典: {motor_name: current_value}，所有关节的实时值
        # Motors dict: {motor_name: current_value}
        self.motors: dict[str, float] = {}
        # 相机 key 列表（按配置顺序）
        # Camera keys in order
        self._camera_keys: list[str] = []

        # 流缓存：后台 gRPC 线程写入的最新值
        # Streaming cache (populated by background threads)
        self._stream_cache: dict[str, Any] = {}
        # 每个缓存值对应的 gRPC 消息头时间戳（独立字典方便拷贝）
        # gRPC header timestamps for cached values (separate dict for clean separation)
        self._stream_ts: dict[str, float] = {}
        # 相机帧环形缓冲区，用于时间戳最近邻对齐
        # Camera frame ring buffer for nearest-neighbor timestamp alignment
        self._camera_ring: dict[str, deque] = {}
        # 环形缓冲区保留帧的最大时长（秒），超时自动裁剪
        # Configurable max history for the camera ring buffer (seconds)
        self._max_camera_history: float = config.max_camera_history
        # 对齐诊断日志相关
        # Alignment diagnostic state
        self._last_alignment_log_time: float = 0.0
        self._alignment_log_interval: float = 2.0  # 日志输出间隔（秒）
        self._stream_lock = threading.Lock()        # 保护所有流缓存的锁
        self._stop_streams = threading.Event()      # 停止后台线程的信号
        self._stream_threads: list[threading.Thread] = []  # 后台线程列表

        self._build_motor_dict()
        self._build_camera_keys()



    def _build_motor_dict(self) -> None:
        """根据配置构建电机字典。"""
        """Build the motors dictionary based on config flags."""
        model = "quanta_x1"

        # 左臂关节
        if self.config.enable_left_arm:
            n_joints = _DEFAULT_JOINT_COUNTS.get(model, {}).get("left_arm", 6)
            for i in range(n_joints):
                self.motors[f"left_arm_{i}.pos"] = 0.0

        # 右臂关节
        if self.config.enable_right_arm:
            n_joints = _DEFAULT_JOINT_COUNTS.get(model, {}).get("right_arm", 6)
            for i in range(n_joints):
                self.motors[f"right_arm_{i}.pos"] = 0.0

        # 夹爪
        if self.config.enable_left_gripper:
            self.motors["left_gripper.pos"] = 0.0
        if self.config.enable_right_gripper:
            self.motors["right_gripper.pos"] = 0.0

        # # 头部（pitch/yaw）
        # if self.config.enable_head:
        #     self.motors["head_0.pos"] = 0.0
        #     self.motors["head_1.pos"] = 0.0

        # # 升降
        # if self.config.enable_lift:
        #     self.motors["lift.pos"] = 0.0

    def _build_camera_keys(self) -> None:
        """根据配置构建相机 key 列表。"""
        if self.config.enable_head_camera:
            self._camera_keys.append("head_camera")
        if self.config.enable_left_arm_camera:
            self._camera_keys.append("left_arm_camera")
        if self.config.enable_right_arm_camera:
            self._camera_keys.append("right_arm_camera")

    def _get_camera_shape(self, camera_key: str) -> tuple[int, int, int]:
        """返回相机图像形状 (H, W, 3)，用于创建黑帧占位。"""
        cfg_map = {
            "head_camera": self.config.head_camera,
            "left_arm_camera": self.config.left_arm_camera,
            "right_arm_camera": self.config.right_arm_camera,
        }
        cfg = cfg_map.get(camera_key)
        if cfg is not None:
            return (cfg.height, cfg.width, 3)
        return (480, 640, 3)  # 默认（找不到配置时）

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征类型字典（全部为 float）。"""
        return {k: float for k in self.motors.keys()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征类型字典（形状元组）。"""
        return {cam: self._get_camera_shape(cam) for cam in self._camera_keys}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """LeRobot 观测特征字典（电机 + 相机）。"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """LeRobot 动作特征字典（仅电机）。"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.is_robot_connected

    @property
    def is_calibrated(self) -> bool:
        """XRobot 不需要外部标定。"""
        return True

    def calibrate(self) -> None:
        """标定（空实现，XRobot 不需要）。"""
        pass

    def connect(self, calibrate: bool = True) -> None:
        """连接到机器人 gRPC 服务，配置控制模式。"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from x2robot import connect as x2_connect
        except ImportError:
            raise ImportError(
                "x2robot package is required for XRobot. Install it with: pip install x2robot"
            )

        self._sdk_robot = x2_connect(f"x2://{self.config.server}")

        logger.info(f"Connected to quanta_x1 at {self.config.server}")

        self.is_robot_connected = True

        # 设置 SDK 控制模式（关节位置 / 末端位姿）
        # Set SDK control mode for action sending
        self.configure()

        logger.info(f"{self} connected.")

    def configure(self) -> None:
        """设置机器人工作模式为 SDK 控制，并选择关节/末端控制模式。"""
        if self._sdk_robot is None:
            return
        try:
            from x2robot.sdk import (
                ManipulatorControlMode,
                ManipulatorControlModeParam,
                RobotModeParam,
                RobotWorkMode,
            )

            # 切换到 SDK 模式（而非示教器模式）
            self._sdk_robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
            if self.config.ctrl_mode == "end_pose":
                # 末端位姿控制模式（位置+姿态）
                self._sdk_robot.robot_control.set_manipulator_control_mode(
                    ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_END_POSE)
                )
            else:
                # 关节位置控制模式（默认）
                self._sdk_robot.robot_control.set_manipulator_control_mode(
                    ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
                )
        except Exception as e:
            logger.warning(f"Failed to configure robot control mode: {e}")



    def _decode_image(self, compressed_image: Any) -> np.ndarray:
        """将 gRPC CompressedImage 解码为 RGB numpy 数组 (H, W, 3)。"""
        """Decode a gRPC CompressedImage to an RGB numpy array (H, W, 3)."""
        import cv2

        data = bytes(compressed_image.data)
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode camera image")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _extract_timestamp_from_header(msg: Any) -> float:
        """从 gRPC 消息头中提取硬件时间戳。

        返回 ``msg.header.stamp`` 的 Unix 时间戳（秒）。
        如果消息没有头部，则回退到 ``time.time()``。
        """
        """Extract gRPC header timestamp (sec + nanosec) as a float seconds.

        Falls back to ``time.time()`` if the message has no header.
        """
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            sec = msg.header.stamp.sec
            nanosec = msg.header.stamp.nanosec
            return float(sec) + float(nanosec) / 1e9
        return time.time()

    # ── gRPC 流式工作线程 ─────────────────────────────────────────────
    # ── gRPC streaming workers ────────────────────────────────────────

    def _camera_stream_worker(self, cam_key: str) -> None:
        """后台线程：读取相机流并缓存最新帧及其时间戳。

        同时维护环形缓冲区 ``_camera_ring``，用于与关节时间戳的最近邻对齐。
        超时 1 秒的旧帧会被自动裁剪。
        """
        """Background thread: read camera stream and cache the latest frame with timestamp."""
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
                    ts = self._extract_timestamp_from_header(frame_msg)
                    with self._stream_lock:
                        self._stream_cache[cam_key] = img
                        self._stream_ts[cam_key] = ts
                        # 维护环形缓冲区
                        if cam_key not in self._camera_ring:
                            self._camera_ring[cam_key] = deque()
                        self._camera_ring[cam_key].append((ts, img))
                        # 裁剪超过 max_camera_history 的旧帧，避免内存膨胀
                        # Prune frames older than max_camera_history
                        cutoff = ts - self._max_camera_history
                        while self._camera_ring[cam_key] and self._camera_ring[cam_key][0][0] < cutoff:
                            self._camera_ring[cam_key].popleft()
                except Exception:
                    logger.debug(f"Failed to decode {cam_key} frame", exc_info=True)
        except Exception as e:
            logger.warning(f"{cam_key} stream stopped: {e}")

    def _gripper_stream_worker(self, side: str) -> None:
        """后台线程：读取夹爪位置流并缓存。

        夹爪不在 ``get_all_joint_states_stream`` 中，
        需要通过独立的 ``get_position_stream`` 读取。
        """
        try:
            gripper = self._sdk_robot.left_gripper if side == "left" else self._sdk_robot.right_gripper
            stream = gripper.get_position_stream(timeout=None)
            motor_key = f"{side}_gripper.pos"
            for msg in stream:
                if self._stop_streams.is_set():
                    break
                if msg is not None:
                    with self._stream_lock:
                        self._stream_cache[motor_key] = float(msg.position)
                        self._stream_ts[motor_key] = time.time()
        except Exception as e:
            logger.warning(f"{side} gripper stream stopped: {e}")

    def _head_joints_stream_worker(self) -> None:
        """后台线程：读取头部关节状态流并缓存。

        头部不在 ``get_all_joint_states_stream`` 中，
        需要通过独立的 ``get_joint_states_stream`` 读取。
        """
        try:
            stream = self._sdk_robot.head.get_joint_states_stream(timeout=None)
            name_mapping: dict[str, str] | None = None
            for state_msg in stream:
                if self._stop_streams.is_set():
                    break
                try:
                    names = list(state_msg.name) if state_msg.name else []
                    positions = list(state_msg.position) if state_msg.position else []
                    if name_mapping is None and names:
                        name_mapping = self._build_joint_name_mapping(names)
                    if name_mapping:
                        ts = self._extract_timestamp_from_header(state_msg)
                        for sdk_name, pos in zip(names, positions):
                            motor_key = name_mapping.get(sdk_name)
                            if motor_key:
                                with self._stream_lock:
                                    self._stream_cache[motor_key] = float(pos)
                                    self._stream_ts[motor_key] = ts
                except Exception:
                    logger.debug("Failed to parse head joint state", exc_info=True)
        except Exception as e:
            logger.warning(f"Head joint state stream stopped: {e}")

    def _build_joint_name_mapping(self, names: list[str]) -> dict[str, str]:
        """构建 SDK 关节名称到内部电机 key 的映射。

        例：'left_arm_joint_1' -> 'left_arm_0.pos'
        使用正则匹配 ``{prefix}_joint_{number}`` 模式，支持下划线分隔的数字。
        如果正则匹配失败，则尝试根据已知电机 key 反向推断。
        """
        """Build mapping from SDK joint names to internal motor keys.

        Example: 'left_arm_joint_1' -> 'left_arm_0.pos'
        """
        import re

        mapping: dict[str, str] = {}
        for name in names:
            # 主模式：{prefix}_joint_{number} 或 {prefix}_joint（无数字时索引为 0）
            # Pattern: {prefix}_joint_{number} or {prefix}_joint
            m = re.match(r"(.+?)_joint_?(\d+)?$", name)
            if m:
                prefix = m.group(1)
                idx_str = m.group(2)
                if idx_str is not None:
                    idx = int(idx_str) - 1          # SDK 从 1 开始，内部从 0 开始
                    key = f"{prefix}_{idx}.pos"
                else:
                    key = f"{prefix}.pos"
                if key in self.motors:
                    mapping[name] = key
                continue

            # 回退策略：根据已知电机 key 反向构建期望的 SDK 名称
            # Fallback: try to match known motor keys
            for motor_key in self.motors:
                base = motor_key.replace(".pos", "")
                parts = base.split("_")
                if parts[-1].isdigit():
                    expected_name = "_".join(parts[:-1]) + "_joint_" + str(int(parts[-1]) + 1)
                    if name == expected_name:
                        mapping[name] = motor_key

        if not mapping:
            logger.warning(f"Could not map joint names to motor keys. Names: {names}, Motors: {list(self.motors.keys())}")
        return mapping

    def _all_joints_stream_worker(self) -> None:
        """后台线程：读取合并后的关节状态流并按名称缓存，附带时间戳。

        从 ``get_all_joint_states_stream`` 获取所有关节的联合状态消息，
        解析每个关节的位置并存储到 ``_stream_cache``，
        消息头的时间戳存储到 ``_stream_ts``。
        """
        """Background thread: read combined joint state stream and cache by name with timestamps."""
        try:
            stream = self._sdk_robot.state.get_all_joint_states_stream(timeout=None)
            name_mapping: dict[str, str] | None = None

            for state_msg in stream:
                if self._stop_streams.is_set():
                    break
                try:
                    names = list(state_msg.name) if state_msg.name else []
                    positions = list(state_msg.position) if state_msg.position else []

                    # 首次收到数据时建立名称映射
                    if name_mapping is None and names:
                        name_mapping = self._build_joint_name_mapping(names)

                    if name_mapping:
                        ts = self._extract_timestamp_from_header(state_msg)
                        for sdk_name, pos in zip(names, positions):
                            motor_key = name_mapping.get(sdk_name)
                            if motor_key:
                                with self._stream_lock:
                                    self._stream_cache[motor_key] = float(pos)
                                    self._stream_ts[motor_key] = ts
                except Exception:
                    logger.debug("Failed to parse joint state", exc_info=True)
        except Exception as e:
            logger.warning(f"Joint state stream stopped: {e}")

    def _start_streams(self) -> None:
        """启动所有后台流线程。

        包括：
        - 1 个合并关节状态流线程（500Hz）
        - N 个相机流线程（每台相机 1 个，~30Hz）
        """
        """Start all background streaming threads."""
        self._stop_streams.clear()

        # 合并关节状态流（500Hz，包含手臂 + 升降）
        # Combined arm + lift joint state stream
        t = threading.Thread(
            target=self._all_joints_stream_worker,
            daemon=True,
            name="stream-all-joints",
        )
        self._stream_threads.append(t)
        t.start()

        # 夹爪独立流（不在合并流中）
        # Gripper streams (not included in combined stream)
        if self.config.enable_left_gripper:
            t = threading.Thread(
                target=self._gripper_stream_worker,
                args=("left",),
                daemon=True,
                name="stream-left-gripper",
            )
            self._stream_threads.append(t)
            t.start()
        if self.config.enable_right_gripper:
            t = threading.Thread(
                target=self._gripper_stream_worker,
                args=("right",),
                daemon=True,
                name="stream-right-gripper",
            )
            self._stream_threads.append(t)
            t.start()

        # 头部独立流（不在合并流中）
        # Head joint stream (not included in combined stream)
        # if self.config.enable_head:
        #     t = threading.Thread(
        #         target=self._head_joints_stream_worker,
        #         daemon=True,
        #         name="stream-head-joints",
        #     )
        #     self._stream_threads.append(t)
        #     t.start()

        # 每个相机独立的流线程（~30Hz）
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

    def start_streams(self) -> None:
        """公开接口：启动相机和关节的后台流线程。"""
        """Start background streaming threads for cameras and joints."""
        self._start_streams()

    def stop_streams(self) -> None:
        """公开接口：停止所有后台流线程。"""
        """Stop all background streaming threads."""
        self._stop_stream_threads()

    def _stop_stream_threads(self) -> None:
        """通知所有流线程停止并等待它们退出（最多 1 秒）。"""
        """Signal all streaming threads to stop and wait for them."""
        self._stop_streams.set()
        for t in self._stream_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._stream_threads.clear()

    def _read_arm_joints(self, side: str) -> list[float] | None:
        """阻塞读取单臂关节位置（不通过流缓存）。失败时返回 None。"""
        """Read joint positions from left or right arm. Returns None on failure."""
        try:
            arm = self._sdk_robot.left_arm if side == "left" else self._sdk_robot.right_arm
            state = arm.get_joint_states()
            # print("关节状态")
            # print(state)
            return list(state.position) if state and state.position else None
        except Exception:
            logger.debug(f"Failed to read {side} arm joint states", exc_info=True)
            return None

    def _read_gripper_position(self, side: str) -> float | None:
        """阻塞读取单侧夹爪位置。失败时返回 None。"""
        """Read gripper position. Returns None on failure."""
        try:
            gripper = self._sdk_robot.left_gripper if side == "left" else self._sdk_robot.right_gripper
            pos = gripper.get_position()
            return float(pos.position) if pos is not None else None
        except Exception:
            logger.debug(f"Failed to read {side} gripper position", exc_info=True)
            return None

    def _read_head_pose(self) -> tuple[float, float] | None:
        """阻塞读取头部姿态 (pitch, yaw)。失败时返回 None。"""
        """Read head pose (pitch, yaw). Returns None on failure."""
        try:
            pose = self._sdk_robot.head.get_pose()
            return (float(pose.pitch), float(pose.yaw))
        except Exception:
            logger.debug("Failed to read head pose", exc_info=True)
            return None

    def _read_lift_position(self) -> float | None:
        """阻塞读取升降位置。失败时返回 None。"""
        try:
            pos = self._sdk_robot.lift.get_lift_position()
            return float(pos.position) if pos is not None else None
        except Exception:
            logger.debug("Failed to read lift position", exc_info=True)
            return None

    def _read_camera_image(self, camera_key: str) -> np.ndarray | None:
        """阻塞读取单帧相机图像。失败时返回 None。"""
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

    def get_observation(self) -> dict[str, Any]:
        """获取当前观测数据，包含对齐后的关节位置和相机图像。

        对齐策略：
        - 参考时间戳取自关节状态流（500Hz，最稳定的硬件时钟）
        - 所有电机 key 来自同一条 gRPC 消息，任意电机的时间戳都具有代表性
        - 对每台相机，在环形缓冲区中按时间戳最近邻匹配到参考时刻的帧
        - 如果环形缓冲区为空，回退到 ``_stream_cache`` 的最新帧

        返回的字典包含 ``timestamp`` 字段，供下游对齐使用。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._stream_lock:
            cache = self._stream_cache.copy()
            ts_cache = self._stream_ts.copy()
            camera_ring = {k: list(v) for k, v in self._camera_ring.items()}

        # 确定参考时间戳，取自关节状态流。
        # 所有电机 key 来自同一条 gRPC 消息，因此任意电机的时间戳都具有代表性。
        # Determine reference timestamp from joint state stream.
        # All motor keys come from the same gRPC message, so any motor's
        # timestamp is representative.
        joint_ts: float | None = None
        for k in self.motors:
            if k in ts_cache:
                joint_ts = ts_cache[k]
                break
        if joint_ts is None:
            joint_ts = time.time()

        # 从默认值开始，然后由流缓存覆盖
        # Start with default values, then override from stream cache
        obs_dict: dict[str, Any] = dict(self.motors)
        for k in self.motors:
            if k in cache:
                obs_dict[k] = cache[k]

        # 每个相机的对齐偏移量（毫秒），用于诊断日志
        # Alignment offsets for diagnostics (ms)
        alignment_offsets: dict[str, float] = {}

        # 填充相机图像，全部对齐到关节参考时间戳（最近邻匹配）
        # Populate camera images, aligned to joint reference timestamp
        for cam_key in self._camera_keys:
            img, cam_ts = self._find_nearest_frame(cam_key, joint_ts, camera_ring)
            if img is not None and cam_ts is not None:
                obs_dict[cam_key] = img
                alignment_offsets[cam_key] = (cam_ts - joint_ts) * 1000.0
            elif cam_key in cache:
                # 回退到最新缓存帧（环形缓冲区可能尚未填充或已被清理）
                # Fall back to latest cached frame
                obs_dict[cam_key] = cache[cam_key]
                if cam_key in ts_cache:
                    alignment_offsets[cam_key] = (ts_cache[cam_key] - joint_ts) * 1000.0
            elif cam_key not in obs_dict:
                # 流尚未产生帧时返回黑帧占位
                # Black frame if stream hasn't produced yet
                obs_dict[cam_key] = np.zeros(self._get_camera_shape(cam_key), dtype=np.uint8)

        # 定期打印对齐统计信息用于诊断
        # Periodic alignment diagnostic logging
        if self.config.log_alignment_stats and alignment_offsets:
            now = time.time()
            if now - self._last_alignment_log_time >= self._alignment_log_interval:
                self._last_alignment_log_time = now
                parts = []
                for cam_key, offset_ms in alignment_offsets.items():
                    parts.append(f"{cam_key}: {offset_ms:+.1f} ms")
                logger.info(
                    "Alignment (camera_ts - joint_ts): %s | ref=%.3f",
                    ", ".join(parts),
                    joint_ts,
                )

        # 更新内部电机状态
        # Update internal motor state
        for k in self.motors:
            if k in obs_dict:
                self.motors[k] = obs_dict[k]

        # 附加参考时间戳，供下游代码对齐使用
        # Expose the reference timestamp for downstream alignment
        obs_dict["timestamp"] = joint_ts

        return obs_dict

    @staticmethod
    def _find_nearest_frame(
        cam_key: str, target_ts: float, camera_ring: dict[str, list[tuple[float, np.ndarray]]],
    ) -> tuple[np.ndarray | None, float | None]:
        """在相机环形缓冲区中查找时间上最接近 *target_ts* 的帧。

        使用 min() 进行 O(n) 最近邻搜索，缓冲区通常 < 30 帧，性能可接受。

        Args:
            cam_key: 相机标识符。
            target_ts: 参考时间戳（秒）。
            camera_ring: 各相机的环形缓冲区快照
                ``{cam_key: [(ts, frame), ...]}``。

        Returns:
            ``(nearest_frame, frame_timestamp)`` 元组；
            如果环形缓冲区为空则返回 ``(None, None)``。
        """
        """Return the camera frame whose timestamp is closest to *target_ts*.

        Returns ``(frame, timestamp)`` or ``(None, None)`` if the ring is empty.
        """
        frames = camera_ring.get(cam_key)
        if not frames:
            return None, None
        best = min(frames, key=lambda item: abs(item[0] - target_ts))
        return best[1], best[0]

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """直接向机器人发送动作指令（含限幅与限速）。

        支持：左臂、右臂、左夹爪、右夹爪、头部、升降。
        对每个组件，如果配置未启用或 action 中无对应 key，则跳过发送。
        action 中不存在的 key 会以 0.0 作为默认值发送（保持安全位置）。

        限幅逻辑：
        - 对每步动作使用 ``max_relative_target`` 限制相对变化量（若配置）
        - 对关节位置和夹爪做绝对值 clamp，防止超出物理限位
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            from x2robot.sdk import GripperPosition, HeadPose, JointPositions
        except ImportError:
            raise ImportError("x2robot package is required for sending actions.")

        # 构建目标位置字典，仅包含当前电机已知的 key
        goal_pos = {k: float(v) for k, v in action.items() if k in self.motors}

        # 限制相对动作幅度（阻止渐进漂移）
        # Limit relative action magnitude
        if self.config.max_relative_target is not None:
            present_pos = {k: float(v) for k, v in self.motors.items() if k in goal_pos}
            goal_present_pos = {k: (goal_pos[k], present_pos[k]) for k in goal_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # # Log left arm joint positions for matplotlib visualization
        # if self.config.enable_left_arm:
        #     _log_file = '/tmp/left_arm_pos_log.txt'
        #     if not hasattr(self, '_left_arm_log_init'):
        #         n_joints = _DEFAULT_JOINT_COUNTS["quanta_x1"].get("left_arm", 6)
        #         header = "step\t" + "\t".join(f"left_arm_{i}.pos" for i in range(n_joints))
        #         with open(_log_file, 'w') as f:
        #             f.write(header + "\n")
        #         self._left_arm_log_init = True
        #     n_joints = _DEFAULT_JOINT_COUNTS["quanta_x1"].get("left_arm", 6)
        #     vals = [float(goal_pos.get(f"left_arm_{i}.pos", 0.0)) for i in range(n_joints)]
        #     step = getattr(self, '_log_step', 0)
        #     self._log_step = step + 1
        #     with open(_log_file, 'a') as f:
        #         f.write(str(step) + "\t" + "\t".join(f"{v:.6f}" for v in vals) + "\n")

        # 发送左臂关节位置（含 clip）
        # Send left arm joint positions (clipped)
        if self.config.enable_left_arm:
            n_joints = _DEFAULT_JOINT_COUNTS["quanta_x1"].get("left_arm", 6)
            positions = [
                float(np.clip(goal_pos.get(f"left_arm_{i}.pos", 0.0), _ARM_LOWER_LIMITS[i], _ARM_UPPER_LIMITS[i]))
                for i in range(n_joints)
            ]
            self._sdk_robot.left_arm.set_joint_positions(JointPositions(positions=positions))

            # # 记录左臂关节位置，便于可视化
            # _clip_log = '/tmp/left_arm_pos_clipped_log.txt'
            # if not hasattr(self, '_left_arm_clip_log_init'):
            #     n_joints = _DEFAULT_JOINT_COUNTS["quanta_x1"].get("left_arm", 6)
            #     header = "step\t" + "\t".join(f"left_arm_{i}.pos" for i in range(n_joints))
            #     with open(_clip_log, 'w') as f:
            #         f.write(header + "\n")
            #     self._left_arm_clip_log_init = True
            # n_joints = _DEFAULT_JOINT_COUNTS["quanta_x1"].get("left_arm", 6)
            # step = getattr(self, '_clip_log_step', 0)
            # self._clip_log_step = step + 1
            # with open(_clip_log, 'a') as f:
            #     f.write(str(step) + "\t" + "\t".join(f"{v:.6f}" for v in positions) + "\n")

        # 发送右臂关节位置（含 clip）
        # Send right arm joint positions (clipped)
        if self.config.enable_right_arm:
            n_joints = _DEFAULT_JOINT_COUNTS["quanta_x1"].get("right_arm", 6)
            positions = [
                float(np.clip(goal_pos.get(f"right_arm_{i}.pos", 0.0), _ARM_LOWER_LIMITS[i], _ARM_UPPER_LIMITS[i]))
                for i in range(n_joints)
            ]
            
            self._sdk_robot.right_arm.set_joint_positions(JointPositions(positions=positions))

        # 发送夹爪位置（阈值 0.5：> 0.5 → 1.5，≤ 0.5 → 0）
        # Send gripper positions
        if self.config.enable_left_gripper and "left_gripper.pos" in self.motors:
            goal = float(goal_pos.get("left_gripper.pos", 0.0))
            pos = 1.5 if goal > 0.5 else 0.0
            self._sdk_robot.left_gripper.set_position(GripperPosition(position=pos))

        if self.config.enable_right_gripper and "right_gripper.pos" in self.motors:
            goal = float(goal_pos.get("right_gripper.pos", 0.0))
            pos = 1.5 if goal > 0.5 else 0.0
            self._sdk_robot.right_gripper.set_position(GripperPosition(position=pos))
            
        # 发送头部姿态（pitch / yaw）
        # Send head pose
        # if self.config.enable_head and "head_0.pos" in self.motors:
        #     pitch = float(goal_pos.get("head_0.pos", 0.0))
        #     yaw = float(goal_pos.get("head_1.pos", 0.0))
        #     self._sdk_robot.head.set_pose(HeadPose(pitch=pitch, yaw=yaw))

        # # 发送升降位置
        # # Send lift position
        # if "lift.pos" in self.motors:
        #     from x2robot.sdk import LiftPosition

        #     pos = float(goal_pos.get("lift.pos", 0.0))
        #     self._sdk_robot.lift.set_lift_position(LiftPosition(position=pos))

        # 更新本地电机状态字典（使用 clamped 后的值）
        self.motors.update(goal_pos)
        return self.motors.copy()

    def get_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """返回当前电机状态的拷贝作为记录的动作。

        参数 ``action`` 保留未使用，与 LeRobot 接口兼容。
        """
        """Return the current motor state as the action (for recording)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.motors.copy()

    @property
    def cameras(self) -> dict:
        """返回相机信息字典，供录制脚本兼容使用。

        每个相机以 ``_CameraInfo`` 对象形式提供 height / width 信息。
        """
        """Compatibility property returning camera info for the record script."""
        result: dict[str, Any] = {}
        for key in self._camera_keys:
            result[key] = type("_CameraInfo", (), {
                "height": self._get_camera_shape(key)[0],
                "width": self._get_camera_shape(key)[1],
            })()
        return result

    def disconnect(self) -> None:
        """断开与机器人的 gRPC 连接，停止所有后台流线程。"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.is_robot_connected = False
        self._stop_stream_threads()
        self._sdk_robot = None

        logger.info(f"{self} disconnected.")
