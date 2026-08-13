"""
完全自定义数据采集示例 - 自动连续录制版本

此脚本保持了原始自定义架构（CustomDataCollector, DataSource, CustomDataFrame），
但将录制方式从"手动逐帧录制"改为 "record.py 风格的自动连续录制"：
- 按回车开始录制，后台自动采集帧数据
- 按回车（或 Ctrl+C）停止录制
- 支持 --hz 控制采集频率
- 支持 --config 选择/自定义数据源

主要特点:
- 保持自定义架构: CustomDataCollector / DataSource / CustomDataFrame
- 自动录制: 后台线程按间隔自动记录帧
- 不依赖封装层: 直接调用机器人接口
- 灵活配置: minimal / vision / full / 自定义数据源

使用方法:
  python custom_data_collection_example_stream.py
  python custom_data_collection_example_stream.py --config minimal
  python custom_data_collection_example_stream.py --config vision
  python custom_data_collection_example_stream.py --config full --hz 15.0
  python custom_data_collection_example_stream.py --config "joint_states,head_rgb" --out ./my_data
"""

import time
import json
import threading
import signal
import sys
import select
import subprocess
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image

from x2robot import connect


@dataclass
class CustomDataFrame:
    """自定义数据帧结构，存储一帧中所有类型的传感器数据"""
    timestamp: float
    frame_id: int

    # 关节状态数据
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_efforts: Optional[np.ndarray] = None

    # 图像数据（多相机）
    images: Dict[str, Image.Image] = None

    # 传感器数据
    imu_data: Optional[Dict[str, Any]] = None           # 惯性测量单元
    odometry: Optional[Dict[str, Any]] = None            # 里程计
    left_arm_end_pose: Optional[Dict[str, Any]] = None   # 左臂末端位姿
    right_arm_end_pose: Optional[Dict[str, Any]] = None  # 右臂末端位姿

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于JSON序列化"""
        data = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
        }

        if self.joint_positions is not None:
            data["joint_positions"] = self.joint_positions.tolist()

        if self.joint_velocities is not None:
            data["joint_velocities"] = self.joint_velocities.tolist()

        if self.joint_efforts is not None:
            data["joint_efforts"] = self.joint_efforts.tolist()

        # 图像数据存储为文件名引用（非二进制）
        if self.images:
            data["images"] = {}
            for cam_name, img in self.images.items():
                data["images"][cam_name] = f"frame_{self.frame_id:04d}_{cam_name}.jpg"

        if self.imu_data:
            data["imu"] = self.imu_data

        if self.odometry:
            data["odometry"] = self.odometry

        if self.left_arm_end_pose:
            data["left_arm_end_pose"] = self.left_arm_end_pose

        if self.right_arm_end_pose:
            data["right_arm_end_pose"] = self.right_arm_end_pose

        return data


class DataSource(Enum):
    """可采集的数据源枚举"""
    JOINT_STATES = "joint_states"              # 关节状态
    HEAD_RGB_CAMERA = "head_rgb"               # 头部RGB相机
    LEFT_ARM_RGB_CAMERA = "left_arm_rgb"       # 左臂RGB相机
    RIGHT_ARM_RGB_CAMERA = "right_arm_rgb"     # 右臂RGB相机
    HEAD_DEPTH_CAMERA = "head_depth"           # 头部深度相机
    CHASSIS_IMU = "imu"                        # IMU传感器
    ODOMETRY = "odometry"                      # 里程计
    LEFT_ARM_END_POSE = "left_arm_end_pose"    # 左臂末端位姿
    RIGHT_ARM_END_POSE = "right_arm_end_pose"  # 右臂末端位姿


class CustomDataCollector:
    """完全自定义数据采集器 - 不使用封装层，直接控制各数据流

    支持两种录制模式:
    - 自动录制（默认）: start_recording() 后自动按 target_hz 记录帧
    - 手动录制: 可随时调用 record_frame() 手动记录
    """

    def __init__(self, robot, output_dir: str = "./custom_collected_data",
                 data_sources: Set[DataSource] = None, target_hz: float = 30.0,
                 use_video_storage: bool = True, image_quality: int = 95):
        """
        初始化自定义数据采集器

        Args:
            robot: 机器人实例
            output_dir: 输出目录
            data_sources: 要采集的数据源集合，如果为None则采集所有数据
            target_hz: 目标采集频率（Hz）
            use_video_storage: 是否使用MP4视频存储图像（True=视频, False=JPG图片）
            image_quality: JPEG图像质量 (1-100)，视频模式下的编码质量
        """
        self.robot = robot
        self.output_dir = Path(output_dir)
        self.data_sources = data_sources or set(DataSource)
        self.target_hz = target_hz
        self.target_period = 1.0 / target_hz
        self.use_video_storage = use_video_storage  # 视频模式: 使用ffmpeg编码为MP4
        self.image_quality = image_quality

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 采集状态
        self.is_collecting = False    # 是否正在采集（数据流运行中）
        self.is_recording = False     # 是否正在录制（保存帧数据）
        self.threads = []             # 采集线程列表
        self.current_episode_data = [] # 当前 episode 的帧数据

        # 关节名称映射
        self.joint_names = None
        self.joint_name_mapping = None

        # 数据缓冲区 - 动态创建各类数据的缓冲列表
        # Data buffer - dynamically created
        self.data_buffer = {}
        self._init_data_buffer()

        # 线程锁（用于线程安全地访问缓冲区）
        self.buffer_lock = threading.Lock()

        # 自动录制线程
        self._record_thread = None
        self._recording_start_time = None
        self._frame_count = 0

    # ========== 新增: 兼容 record.py 风格接口 ==========

    @property
    def episode_count(self) -> int:
        """已保存的 episode 数量（兼容 DataCollector 接口）"""
        return len(list(self.output_dir.glob("episode_*")))

    def print_stats(self):
        """打印当前采集统计信息（兼容 DataCollector 接口）"""
        if not self.is_recording:
            return

        elapsed = time.time() - self._recording_start_time if self._recording_start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        parts = [f"Frames: {self._frame_count} | Elapsed: {elapsed:.1f}s | FPS: {fps:.1f}"]

        with self.buffer_lock:
            if 'joint_states' in self.data_buffer:
                parts.append(f"Joint: {len(self.data_buffer['joint_states'])}")
            for cam_name, buf in self.data_buffer.get('images', {}).items():
                if buf:
                    parts.append(f"{cam_name}: {len(buf)}")

        print(f"  {' | '.join(parts)}", end='\r')

    def print_sensor_fps(self, window_seconds: float = 5.0):
        """输出各传感器数据流的实际帧率信息

        基于缓冲区中各数据源的时间戳，计算最近 window_seconds 秒内的实际 FPS。

        Args:
            window_seconds: 统计窗口大小（秒），默认 5 秒
        """
        if not self.is_recording and not self.is_collecting:
            print("（未在采集状态，无帧率数据）")
            return

        now = time.time()
        cutoff = now - window_seconds

        print(f"\n{'='*60}")
        print(f"  传感器帧率统计 (最近 {window_seconds:.0f}s 窗口)")
        print(f"{'='*60}")
        print(f"  {'数据源':<24} {'帧数':>6}  {'帧率 (Hz)':>10}")
        print(f"  {'-'*42}")

        with self.buffer_lock:
            # -- 关节状态 --
            if 'joint_states' in self.data_buffer:
                buf = self.data_buffer['joint_states']
                recent = [x for x in buf if x['timestamp'] >= cutoff]
                count = len(recent)
                fps_val = count / window_seconds if window_seconds > 0 else 0
                print(f"  {'joint_states':<24} {count:>6}  {fps_val:>10.1f}")

            # -- 图像数据（各路相机） --
            for cam_name, buf in sorted(self.data_buffer.get('images', {}).items()):
                recent = [x for x in buf if x['timestamp'] >= cutoff]
                count = len(recent)
                fps_val = count / window_seconds if window_seconds > 0 else 0
                print(f"  {cam_name:<24} {count:>6}  {fps_val:>10.1f}")

            # -- IMU --
            if 'imu' in self.data_buffer:
                buf = self.data_buffer['imu']
                recent = [x for x in buf if x['timestamp'] >= cutoff]
                count = len(recent)
                fps_val = count / window_seconds if window_seconds > 0 else 0
                print(f"  {'imu':<24} {count:>6}  {fps_val:>10.1f}")

            # -- 里程计 --
            if 'odometry' in self.data_buffer:
                buf = self.data_buffer['odometry']
                recent = [x for x in buf if x['timestamp'] >= cutoff]
                count = len(recent)
                fps_val = count / window_seconds if window_seconds > 0 else 0
                print(f"  {'odometry':<24} {count:>6}  {fps_val:>10.1f}")

            # -- 手臂末端位姿 --
            for key in ['left_arm_end_pose', 'right_arm_end_pose']:
                if key in self.data_buffer:
                    buf = self.data_buffer[key]
                    recent = [x for x in buf if x['timestamp'] >= cutoff]
                    count = len(recent)
                    fps_val = count / window_seconds if window_seconds > 0 else 0
                    label = key.replace('_', ' ')
                    print(f"  {label:<24} {count:>6}  {fps_val:>10.1f}")

        print(f"{'='*60}\n")

    # ========== 缓冲区初始化 ==========

    def _init_data_buffer(self):
        """根据启用的数据源初始化数据缓冲区"""
        # 关节状态缓冲区
        if DataSource.JOINT_STATES in self.data_sources:
            self.data_buffer['joint_states'] = []

        # 图像数据缓冲区（按相机名称分组的字典）
        self.data_buffer['images'] = {}
        camera_sources = [DataSource.HEAD_RGB_CAMERA, DataSource.LEFT_ARM_RGB_CAMERA,
                         DataSource.RIGHT_ARM_RGB_CAMERA, DataSource.HEAD_DEPTH_CAMERA]
        for cam_source in camera_sources:
            if cam_source in self.data_sources:
                self.data_buffer['images'][cam_source.value] = []

        # 传感器数据缓冲区
        sensor_sources = [DataSource.CHASSIS_IMU, DataSource.ODOMETRY,
                         DataSource.LEFT_ARM_END_POSE, DataSource.RIGHT_ARM_END_POSE]
        for sensor_source in sensor_sources:
            if sensor_source in self.data_sources:
                self.data_buffer[sensor_source.value] = []

    def _extract_timestamp_from_header(self, msg):
        """从ROS消息的header中提取时间戳（秒）"""
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            sec = msg.header.stamp.sec
            nanosec = msg.header.stamp.nanosec
            return float(sec) + float(nanosec) / 1e9
        else:
            return time.time()

    # ========== 数据采集线程方法 ==========

    def _collect_joint_states(self):
        """采集关节状态数据（独立线程）"""
        print("启动关节状态采集...")
        try:
            # 获取关节状态流（阻塞式）
            stream = self.robot.state.get_all_joint_states_stream(timeout=None)

            for state_msg in stream:
                if not self.is_collecting:
                    break

                try:
                    # 首次收到消息时建立关节名称映射
                    # Build joint name mapping on first message
                    if self.joint_name_mapping is None and hasattr(state_msg, 'name') and state_msg.name:
                        self.joint_names = list(state_msg.name)
                        self.joint_name_mapping = {name: idx for idx, name in enumerate(state_msg.name)}
                        print(f"关节名称: {self.joint_names}")

                    # 提取关节位置、速度、力矩数据
                    joint_positions = np.array(state_msg.position, dtype=np.float32)
                    joint_velocities = np.array(state_msg.velocity, dtype=np.float32) if hasattr(state_msg, 'velocity') and state_msg.velocity else None
                    joint_efforts = np.array(state_msg.effort, dtype=np.float32) if hasattr(state_msg, 'effort') and state_msg.effort else None

                    timestamp = self._extract_timestamp_from_header(state_msg)

                    # 线程安全地写入缓冲区
                    with self.buffer_lock:
                        self.data_buffer['joint_states'].append({
                            'timestamp': timestamp,
                            'positions': joint_positions,
                            'velocities': joint_velocities,
                            'efforts': joint_efforts
                        })

                    # 简单限流：约100Hz采集
                    # Simple rate limiting: ~100Hz
                    time.sleep(0.01)

                except Exception as e:
                    print(f"关节状态处理错误: {e}")
                    continue

        except Exception as e:
            print(f"关节状态流错误: {e}")

    def _collect_camera_stream(self, camera_name, stream_func):
        """采集指定相机视频流（独立线程）"""
        print(f"启动 {camera_name} 采集...")

        try:
            stream = stream_func(timeout=None)

            for frame_msg in stream:
                if not self.is_collecting:
                    break

                try:
                    if not frame_msg or not frame_msg.data:
                        continue

                    # 解码图像字节为PIL Image
                    img_bytes = bytes(frame_msg.data)
                    img = Image.open(io.BytesIO(img_bytes))

                    # 统一转换为 RGB 模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    timestamp = self._extract_timestamp_from_header(frame_msg)

                    with self.buffer_lock:
                        if camera_name not in self.data_buffer['images']:
                            self.data_buffer['images'][camera_name] = []
                        self.data_buffer['images'][camera_name].append({
                            'timestamp': timestamp,
                            'image': img
                        })

                    # 限流：约30Hz
                    # Rate limiting: ~30Hz
                    time.sleep(0.033)

                except Exception as e:
                    print(f"{camera_name} 处理错误: {e}")
                    continue

        except Exception as e:
            print(f"{camera_name} 流错误: {e}")

    def _collect_imu(self):
        """采集IMU数据（独立线程）- 获取姿态、角速度、加速度"""
        print("启动IMU采集...")
        try:
            stream = self.robot.imu.get_chassis_imu_stream(timeout=None)

            for imu_msg in stream:
                if not self.is_collecting:
                    break

                try:
                    # 检查消息有效性
                    if not imu_msg:
                        continue

                    # 构建IMU数据结构
                    imu_data = {
                        'orientation': [          # 姿态（四元数）
                            float(imu_msg.orientation.x) if imu_msg.orientation else 0,
                            float(imu_msg.orientation.y) if imu_msg.orientation else 0,
                            float(imu_msg.orientation.z) if imu_msg.orientation else 0,
                            float(imu_msg.orientation.w) if imu_msg.orientation else 1
                        ],
                        'angular_velocity': [     # 角速度（rad/s）
                            float(imu_msg.angular_velocity.x) if imu_msg.angular_velocity else 0,
                            float(imu_msg.angular_velocity.y) if imu_msg.angular_velocity else 0,
                            float(imu_msg.angular_velocity.z) if imu_msg.angular_velocity else 0
                        ],
                        'linear_acceleration': [  # 线加速度（m/s²）
                            float(imu_msg.linear_acceleration.x) if imu_msg.linear_acceleration else 0,
                            float(imu_msg.linear_acceleration.y) if imu_msg.linear_acceleration else 0,
                            float(imu_msg.linear_acceleration.z) if imu_msg.linear_acceleration else 0
                        ]
                    }

                    timestamp = self._extract_timestamp_from_header(imu_msg)

                    with self.buffer_lock:
                        self.data_buffer['imu'].append({
                            'timestamp': timestamp,
                            'data': imu_data
                        })

                    time.sleep(0.1)  # 10Hz采集

                except Exception as e:
                    print(f"IMU处理错误: {e}")
                    continue

        except Exception as e:
            print(f"IMU流错误: {e}")

    def _collect_odometry(self):
        """采集里程计数据（独立线程）- 获取位姿和速度"""
        print("启动里程计采集...")
        try:
            stream = self.robot.chassis.get_odometry_stream(timeout=None)

            for odom_msg in stream:
                if not self.is_collecting:
                    break

                try:
                    # 检查消息有效性
                    if not odom_msg:
                        continue

                    # 安全提取里程计数据（含默认值处理）
                    odometry_data = {
                        'pose': {
                            'position': {
                                'x': odom_msg.pose.pose.position.x if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.position else 0,
                                'y': odom_msg.pose.pose.position.y if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.position else 0,
                                'z': odom_msg.pose.pose.position.z if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.position else 0,
                            },
                            'orientation': {
                                'x': odom_msg.pose.pose.orientation.x if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.orientation else 0,
                                'y': odom_msg.pose.pose.orientation.y if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.orientation else 0,
                                'z': odom_msg.pose.pose.orientation.z if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.orientation else 0,
                                'w': odom_msg.pose.pose.orientation.w if odom_msg.pose and odom_msg.pose.pose and odom_msg.pose.pose.orientation else 1,
                            }
                        },
                        'twist': {
                            'linear': {
                                'x': odom_msg.twist.twist.linear.x if odom_msg.twist and odom_msg.twist.twist and odom_msg.twist.twist.linear else 0,
                                'y': odom_msg.twist.twist.linear.y if odom_msg.twist and odom_msg.twist.twist and odom_msg.twist.twist.linear else 0,
                                'z': odom_msg.twist.twist.linear.z if odom_msg.twist and odom_msg.twist.twist and odom_msg.twist.twist.linear else 0,
                            },
                            'angular': {
                                'x': odom_msg.twist.twist.angular.x if odom_msg.twist and odom_msg.twist.twist and odom_msg.twist.twist.angular else 0,
                                'y': odom_msg.twist.twist.angular.y if odom_msg.twist and odom_msg.twist.twist and odom_msg.twist.twist.angular else 0,
                                'z': odom_msg.twist.twist.angular.z if odom_msg.twist and odom_msg.twist.twist and odom_msg.twist.twist.angular else 0,
                            }
                        }
                    }

                    timestamp = self._extract_timestamp_from_header(odom_msg)

                    with self.buffer_lock:
                        self.data_buffer['odometry'].append({
                            'timestamp': timestamp,
                            'data': odometry_data
                        })

                    time.sleep(0.1)  # 10Hz采集

                except Exception as e:
                    print(f"里程计处理错误: {e}")
                    continue

        except Exception as e:
            print(f"里程计流错误: {e}")

    def _collect_arm_end_pose(self, arm_name):
        """采集手臂末端位姿（独立线程）

        Args:
            arm_name: 手臂名称（'left' 或 'right'）
        """
        print(f"启动{arm_name}末端位姿采集...")

        # 根据手臂名称选择对应的对象和流方法
        # Arm object name mapping
        arm_attr_name = f"{arm_name}_arm"
        stream_func = getattr(self.robot, arm_attr_name).get_end_pose_stream
        buffer_key = f"{arm_name}_arm_end_pose"

        try:
            stream = stream_func(timeout=None)

            for pose_msg in stream:
                if not self.is_collecting:
                    break

                try:
                    # 检查消息有效性
                    if not pose_msg:
                        continue

                    # 提取位姿数据（位置 + 姿态四元数）
                    pose_data = {
                        'position': {
                            'x': pose_msg.pose.position.x if pose_msg.pose and pose_msg.pose.position else 0,
                            'y': pose_msg.pose.position.y if pose_msg.pose and pose_msg.pose.position else 0,
                            'z': pose_msg.pose.position.z if pose_msg.pose and pose_msg.pose.position else 0,
                        },
                        'orientation': {
                            'x': pose_msg.pose.orientation.x if pose_msg.pose and pose_msg.pose.orientation else 0,
                            'y': pose_msg.pose.orientation.y if pose_msg.pose and pose_msg.pose.orientation else 0,
                            'z': pose_msg.pose.orientation.z if pose_msg.pose and pose_msg.pose.orientation else 0,
                            'w': pose_msg.pose.orientation.w if pose_msg.pose and pose_msg.pose.orientation else 1,
                        }
                    }

                    timestamp = self._extract_timestamp_from_header(pose_msg)

                    with self.buffer_lock:
                        self.data_buffer[buffer_key].append({
                            'timestamp': timestamp,
                            'data': pose_data
                        })

                    time.sleep(0.033)  # ~30Hz采集

                except Exception as e:
                    print(f"{arm_name}末端位姿处理错误: {e}")
                    continue

        except Exception as e:
            print(f"{arm_name}末端位姿流错误: {e}")

    def start_collecting(self):
        """启动所有数据采集线程"""
        if self.is_collecting:
            print("已在采集中")
            return

        self.is_collecting = True

        # 清空所有数据缓冲区
        # Clear all buffers
        with self.buffer_lock:
            for key in self.data_buffer:
                if isinstance(self.data_buffer[key], list):
                    self.data_buffer[key].clear()
                elif isinstance(self.data_buffer[key], dict):
                    for sub_key in self.data_buffer[key]:
                        self.data_buffer[key][sub_key].clear()

        # 创建采集线程列表
        threads = []

        # 关节状态采集线程
        if DataSource.JOINT_STATES in self.data_sources:
            threads.append(threading.Thread(target=self._collect_joint_states, daemon=True))

        # 相机流采集线程（头部RGB、左/右臂、头部深度）
        camera_configs = [
            (DataSource.HEAD_RGB_CAMERA, 'head_rgb', self.robot.head_camera.get_rgb_video_stream),
            (DataSource.LEFT_ARM_RGB_CAMERA, 'left_arm_rgb', self.robot.left_arm_camera.get_video_stream),
            (DataSource.RIGHT_ARM_RGB_CAMERA, 'right_arm_rgb', self.robot.right_arm_camera.get_video_stream),
            (DataSource.HEAD_DEPTH_CAMERA, 'head_depth', self.robot.head_camera.get_depth_video_stream),
        ]

        for data_source, cam_name, stream_func in camera_configs:
            if data_source in self.data_sources:
                threads.append(threading.Thread(
                    target=self._collect_camera_stream,
                    args=(cam_name, stream_func),
                    daemon=True
                ))

        # 传感器采集线程
        if DataSource.CHASSIS_IMU in self.data_sources:
            threads.append(threading.Thread(target=self._collect_imu, daemon=True))

        if DataSource.ODOMETRY in self.data_sources:
            threads.append(threading.Thread(target=self._collect_odometry, daemon=True))

        # 末端位姿采集线程
        if DataSource.LEFT_ARM_END_POSE in self.data_sources:
            threads.append(threading.Thread(
                target=self._collect_arm_end_pose,
                args=('left',),
                daemon=True
            ))

        if DataSource.RIGHT_ARM_END_POSE in self.data_sources:
            threads.append(threading.Thread(
                target=self._collect_arm_end_pose,
                args=('right',),
                daemon=True
            ))

        # 启动所有线程
        for t in threads:
            t.start()
            self.threads.append(t)

        # 显示启用的数据源
        enabled_sources = [src.value for src in self.data_sources]
        print(f"✓ 数据采集已启动 (目标频率: {self.target_hz} Hz)")
        print(f"  输出目录: {self.output_dir}")
        print(f"  图像存储: {'MP4视频' if self.use_video_storage else 'JPG图片'}")
        print(f"  启用的数据源: {enabled_sources}")

    def stop_collecting(self):
        """停止所有数据采集线程"""
        self.is_collecting = False

        # 等待所有采集线程结束
        for t in self.threads:
            t.join(timeout=2.0)

        self.threads.clear()
        print("✓ 数据采集已停止")

    # ========== 自动录制（新增: record.py 风格） ==========

    def _auto_record_loop(self):
        """自动录制后台线程: 按 target_hz 频率持续调用 record_frame()"""
        last_frame_time = 0
        while self.is_recording and self.is_collecting:
            loop_start = time.time()
            self.record_frame()
            self._frame_count += 1
            elapsed = time.time() - loop_start
            sleep_time = self.target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start_recording(self, task: str = "custom_task"):
        """开始录制 episode - 自动模式（兼容 record.py 风格）

        自动启动后台线程持续记录帧，直到调用 stop_recording()。

        Args:
            task: 任务名称
        """
        if not self.is_collecting:
            raise RuntimeError("请先调用 start_collecting()")

        self.is_recording = True
        self.current_episode_data = []
        self._recording_start_time = time.time()
        self._frame_count = 0

        # 启动自动录制后台线程
        self._record_thread = threading.Thread(target=self._auto_record_loop, daemon=True)
        self._record_thread.start()

        print(f"  Recording... press Enter to STOP\n")

    def record_frame(self):
        """记录一帧数据 - 从各缓冲区中获取最新的数据快照

        此方法可被自动录制线程调用，也可被手动调用。
        """
        if not self.is_recording:
            return None

        # 获取当前时间戳
        current_time = time.time()

        # 线程安全地从缓冲区获取最近的数据
        with self.buffer_lock:
            frame_data = CustomDataFrame(
                timestamp=current_time,
                frame_id=len(self.current_episode_data)
            )

            # 获取最新的关节状态
            if DataSource.JOINT_STATES in self.data_sources and self.data_buffer.get('joint_states'):
                joint_data = self.data_buffer['joint_states'][-1]
                frame_data.joint_positions = joint_data['positions']
                frame_data.joint_velocities = joint_data['velocities']
                frame_data.joint_efforts = joint_data['efforts']

            # 获取各部相机的最新图像
            for cam_name, cam_buffer in self.data_buffer.get('images', {}).items():
                if cam_buffer:
                    frame_data.images = frame_data.images or {}
                    frame_data.images[cam_name] = cam_buffer[-1]['image']  # 最新的图像

            # 获取各传感器的最新数据
            if DataSource.CHASSIS_IMU in self.data_sources and self.data_buffer.get('imu'):
                frame_data.imu_data = self.data_buffer['imu'][-1]['data']

            if DataSource.ODOMETRY in self.data_sources and self.data_buffer.get('odometry'):
                frame_data.odometry = self.data_buffer['odometry'][-1]['data']

            if DataSource.LEFT_ARM_END_POSE in self.data_sources and self.data_buffer.get('left_arm_end_pose'):
                frame_data.left_arm_end_pose = self.data_buffer['left_arm_end_pose'][-1]['data']

            if DataSource.RIGHT_ARM_END_POSE in self.data_sources and self.data_buffer.get('right_arm_end_pose'):
                frame_data.right_arm_end_pose = self.data_buffer['right_arm_end_pose'][-1]['data']

        self.current_episode_data.append(frame_data)
        return frame_data

    def _collect_camera_frames(self):
        """从当前 episode 的帧数据中提取各相机的图像序列"""
        camera_frames = {}
        for frame in self.current_episode_data:
            if frame.images:
                for cam_name, img in frame.images.items():
                    if cam_name not in camera_frames:
                        camera_frames[cam_name] = []
                    camera_frames[cam_name].append(img)
        return camera_frames

    def _create_video_with_ffmpeg(self, images: list, output_path: str, fps: float):
        """使用 ffmpeg 将 PIL Image 列表编码为 MP4 视频

        Args:
            images: PIL Image 列表
            output_path: 输出 MP4 文件路径
            fps: 视频帧率

        Returns:
            是否成功
        """
        if not images:
            return False

        try:
            # ffmpeg 命令: 从 stdin 读取 MJPEG 流, 编码为 H.264 MP4
            ffmpeg_cmd = [
                "ffmpeg",
                "-r", str(fps),
                "-f", "image2pipe",      # 从 stdin 管道读取图像
                "-vcodec", "mjpeg",       # 输入格式: MJPEG
                "-pix_fmt", "yuvj420p",
                "-i", "-",                # stdin
                "-c:v", "libx264",         # 输出编码: H.264
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", str(max(1, min(51, 100 - self.image_quality))),  # 质量: 高质量→低CRF
                "-y",                      # 覆盖已有文件
                str(output_path)
            ]

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            # 逐帧将 PIL Image 编码为 JPEG 字节并写入 ffmpeg stdin
            for img in images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=self.image_quality)
                process.stdin.write(buf.getvalue())

            process.stdin.close()
            process.wait(timeout=60)

            if process.returncode != 0:
                stderr = process.stderr.read().decode()
                print(f"  ⚠️  ffmpeg 错误: {stderr[:200]}")
                return False
            return True
        except Exception as e:
            print(f"  ⚠️  视频编码失败: {e}")
            return False

    def stop_recording(self, task: str = "custom_task") -> Dict[str, Any]:
        """停止录制并保存所有数据到磁盘

        视频模式下: 使用 ffmpeg 将图像帧编码为 MP4，JSON 中存储帧索引
        图片模式下: 每帧保存为独立 JPEG，JSON 中存储文件路径

        Args:
            task: 任务名称

        Returns:
            包含 episode 信息的字典，格式与 DataCollector 兼容
        """
        if not self.is_recording:
            return None

        self.is_recording = False

        # 等待自动录制线程结束
        if self._record_thread is not None:
            self._record_thread.join(timeout=3.0)
            self._record_thread = None

        recording_duration = time.time() - self._recording_start_time
        num_frames = len(self.current_episode_data)

        # 创建 episode 目录（自动编号）
        episode_id = len(list(self.output_dir.glob("episode_*")))
        episode_dir = self.output_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir(exist_ok=True)

        # 构建 episode 元数据
        episode_data = {
            "episode_id": episode_id,
            "task": task,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": recording_duration,
            "num_frames": num_frames,
            "joint_names": self.joint_names or [],
            "storage_format": "video" if self.use_video_storage else "images",
            "frames": []
        }

        # -- 视频模式 --
        if self.use_video_storage and self.current_episode_data:
            # 收集所有帧中各相机的图像序列
            camera_frames = self._collect_camera_frames()
            video_files = {}

            print(f"  正在编码视频 ({len(camera_frames)} 路相机)...")
            for cam_name, images in camera_frames.items():
                # 跳过 depth 相机（深度图需特殊处理）
                if "depth" in cam_name.lower():
                    # 深度相机图片仍以 JPEG 保存
                    for frame in self.current_episode_data:
                        if frame.images and cam_name in frame.images:
                            img = frame.images[cam_name]
                            frame_id = frame.frame_id
                            img_filename = f"frame_{frame_id:04d}_{cam_name}.jpg"
                            img_path = episode_dir / img_filename
                            img.save(img_path, 'JPEG', quality=self.image_quality)
                else:
                    video_path = episode_dir / f"{cam_name}.mp4"
                    self._create_video_with_ffmpeg(images, str(video_path), self.target_hz)
                    video_files[cam_name] = f"{cam_name}.mp4"

            if video_files:
                episode_data["video_files"] = video_files

            # 构建帧数据：视频模式下 images 存储整数帧索引
            for frame in self.current_episode_data:
                frame_dict = frame.to_dict()
                if frame.images and video_files:
                    frame_dict["images"] = {}
                    for cam_name in frame.images:
                        if cam_name in video_files:
                            # 视频模式: 存储帧索引（整数）
                            frame_dict["images"][cam_name] = frame.frame_id
                        else:
                            # 深度相机等非视频: 保持文件名
                            frame_dict["images"][cam_name] = f"frame_{frame.frame_id:04d}_{cam_name}.jpg"
                episode_data["frames"].append(frame_dict)

        # -- 图片模式（原有逻辑） --
        else:
            for frame in self.current_episode_data:
                frame_dict = frame.to_dict()
                # 保存图像文件到磁盘
                if frame.images:
                    for cam_name, img in frame.images.items():
                        img_filename = f"frame_{frame.frame_id:04d}_{cam_name}.jpg"
                        img_path = episode_dir / img_filename
                        img.save(img_path, 'JPEG', quality=self.image_quality)
                episode_data["frames"].append(frame_dict)

        # 保存 JSON 元数据文件
        json_path = episode_dir / "episode.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved Episode {episode_id}: {num_frames} frames, {recording_duration:.2f}s")
        print(f"  保存路径: {episode_dir}")

        # 清空当前 episode 数据
        self.current_episode_data = []
        self._frame_count = 0

        return {
            "episode_id": episode_id,
            "episode_dir": str(episode_dir),
            "num_frames": num_frames,
            "duration": recording_duration
        }


# ========== 数据源预设配置 ==========

def create_minimal_data_sources() -> Set[DataSource]:
    """创建最小化数据采集配置（只采集关节状态）"""
    return {DataSource.JOINT_STATES}


def create_full_data_sources() -> Set[DataSource]:
    """创建完整数据采集配置（关节状态 + 三路RGB相机 + 双臂末端位姿）"""
    return {
        DataSource.JOINT_STATES,
        DataSource.HEAD_RGB_CAMERA,
        DataSource.LEFT_ARM_RGB_CAMERA,
        DataSource.RIGHT_ARM_RGB_CAMERA,
        DataSource.LEFT_ARM_END_POSE,
        DataSource.RIGHT_ARM_END_POSE,
    }


def create_vision_only_sources() -> Set[DataSource]:
    """创建仅视觉数据采集配置（所有相机的RGB和深度数据）"""
    return {
        DataSource.HEAD_RGB_CAMERA,
        DataSource.LEFT_ARM_RGB_CAMERA,
        DataSource.RIGHT_ARM_RGB_CAMERA,
        DataSource.HEAD_DEPTH_CAMERA,
    }


# ========== 主函数（record.py 风格） ==========

def main(
    server: str = "localhost:50051",
    config: str = "full",  # minimal（最小化）, full（完整）, vision（仅视觉）或逗号分隔的自定义数据源
    out: str = "./custom_collected_data",  # 输出目录（兼容 --out / --output-dir）
    hz: float = 30.0,  # 目标录制频率（Hz）
    use_video_storage: bool = True,  # 是否使用MP4视频存储（默认True，与record.py一致）
    image_quality: int = 95,  # JPEG/视频编码质量 (1-100)
):
    """
    主函数 - 自定义数据采集入口（record.py 风格自动录制）

    Args:
        server: 机器人服务器地址 (host:port)
        config: 采集配置 preset (minimal/full/vision) 或逗号分隔的数据源列表
        out: 输出目录
        hz: 目标采集频率（Hz）
        use_video_storage: 是否使用MP4视频存储（默认True）
        image_quality: JPEG/视频编码质量 (1-100)
    """
    # 注册信号处理器（处理 Ctrl+C）
    def signal_handler(sig, frame):
        print("\n\n收到中断信号，正在停止...")
        if 'collector' in globals():
            collector.stop_collecting()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 根据配置选择数据源
    if config == "minimal":
        data_sources = create_minimal_data_sources()
    elif config == "vision":
        data_sources = create_vision_only_sources()
    elif config == "full":
        data_sources = create_full_data_sources()
    else:
        # 自定义配置：解析逗号分隔的数据源名称
        data_sources = set()
        config_parts = config.split(',')
        for part in config_parts:
            part = part.strip()
            try:
                data_sources.add(DataSource(part))
            except ValueError:
                print(f"警告: 未知的数据源 '{part}'，已忽略")
                continue

    print("将要采集的数据源:")
    for source in sorted(data_sources, key=lambda x: x.value):
        print(f"  - {source.value}")
    print()

    # 连接机器人
    print(f"正在连接机器人 {server}...")
    robot = connect(f"x2://{server}")
    print(f"✓ 机器人连接成功 ({robot.get_robot_model()})\n")

    # 创建自定义采集器
    collector = CustomDataCollector(
        robot=robot,
        output_dir=out,
        data_sources=data_sources,
        target_hz=hz,
        use_video_storage=use_video_storage,
        image_quality=image_quality,
    )

    # 启动数据采集线程
    collector.start_collecting()
    print(f"Output: {collector.output_dir}")
    print(f"Target FPS: {collector.target_hz} Hz\n")

    try:
        # 录制多个 episodes（record.py 风格循环）
        while True:
            current_ep = collector.episode_count
            print(f"\n=== Episode {current_ep} ===")
            # 输入任务名称，默认为 "default"
            task = input("Task name (Enter for 'default'): ").strip() or "default"
            # 按回车键开始录制
            input("Press Enter to START recording...")

            # 开始自动录制（启动后台线程持续记录帧）
            collector.start_recording(task=task)
            print("Recording... press Enter to STOP\n")

            # 使用 select 实现非阻塞的按回车停止等待（record.py 风格）
            # 每 1 秒检查一次是否有键盘输入，同时打印采集统计信息
            try:
                while True:
                    if select.select([sys.stdin], [], [], 1)[0]:
                        input()  # 读取回车键
                        break
                    collector.print_stats()
            except KeyboardInterrupt:
                pass

            # 停止录制并保存
            if collector.is_recording:
                info = collector.stop_recording(task=task)
                if info:
                    print(f"\nSaved Episode {info['episode_id']}: {info['num_frames']} frames, {info['duration']:.2f}s")

            # 询问是否继续录制
            cont = input("\nContinue? (y/n, default n): ").strip().lower()
            if cont != "y":
                break

    except KeyboardInterrupt:
        print("\n用户中断录制")
    finally:
        # 停止数据采集线程
        collector.stop_collecting()
        print(f"\nDone. {collector.episode_count} episodes in {collector.output_dir}")


if __name__ == "__main__":
    import typer
    typer.run(main)
