"""
Sensor Data Collection Configuration

Define all collectible sensor data streams and configuration options
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CollectionConfig:
    """Sensor collection configuration\n传感器采集配置"""

    # Joint states / 关节状态
    slave_joint_names: Optional[List[str]] = None
    """从臂关节状态名称列表
    Slave arm joint state name list, e.g.: ['left_arm_joint_states', 'right_arm_joint_states']
    If None or empty list, slave arm joint states are not collected
    Supported joint state names:
    - left_arm_joint_states: 左臂关节状态
    - right_arm_joint_states: 右臂关节状态
    - lift_joint_states: 升降关节状态 (quanta_x1)
    - waist_joint_states: 腰部关节状态 (quanta_x2)
    - left_gripper_joint_states: 左夹爪关节状态
    - right_gripper_joint_states: 右夹爪关节状态
    - head_joint_states: 头部关节状态"""
    
    slave_action_names: Optional[List[str]] = None
    """从臂动作名称列表（自动生成时替换 _joint_states 为 _actions）
    Slave arm action name list, e.g.: ['left_arm_actions', 'right_arm_actions']
    If None, will be automatically generated from slave_joint_names (replace '_joint_states' with '_actions')
    If [], no action data will be collected"""
    
    # Image sensors / 图像传感器（4 路固定）
    enable_head_rgb_stream: bool = False
    """启用头部 RGB 视频流"""

    enable_head_depth_stream: bool = False
    """启用头部深度视频流"""

    enable_left_arm_rgb_stream: bool = False
    """启用左臂 RGB 视频流"""

    enable_right_arm_rgb_stream: bool = False
    """启用右臂 RGB 视频流"""
    
    # End pose / 末端位姿
    enable_left_arm_end_pose: bool = False
    """启用左臂末端位姿"""
    
    enable_right_arm_end_pose: bool = False
    """启用右臂末端位姿"""

    enable_wrench_ext_world: bool = False
    """启用腕部外力（世界坐标系）"""

    enable_wrench_ext_local: bool = False
    """启用腕部外力（局部坐标系）"""

    enable_waist_end_pose: bool = False
    """启用腰部末端位姿"""

    enable_odometry: bool = False
    """启用底盘里程计 (odom)"""
    
    enable_pose: bool = False
    """启用机器人位姿数据 (tracked_pose)"""
    
    enable_chassis_imu: bool = False
    """启用底盘 IMU 数据"""
    
    # Depth sensors / 深度传感器
    enable_depth_points: bool = False
    """启用底盘深度点云"""
    
    enable_head_depth_video: bool = False
    """启用头部深度视频流"""
    
    # Laser scanner / 激光扫描
    enable_laser_scan: bool = False
    """启用激光扫描仪"""

    enable_left_gripper_position: bool = False
    """启用左夹爪位置"""

    enable_right_gripper_position: bool = False
    """启用右夹爪位置"""
    
    # Tactile sensors / 触觉传感器
    enable_left_gripper_tactile: bool = False
    """启用左夹爪触觉传感器"""
    
    enable_right_gripper_tactile: bool = False
    """启用右夹爪触觉传感器"""
    
    enable_left_hand_tactile: bool = False
    """启用左手触觉传感器"""
    
    enable_right_hand_tactile: bool = False
    """启用右手触觉传感器"""
    
    # 距离传感器
    enable_tof_sensors: bool = False
    """启用 ToF 传感器 (2 个)"""
    
    enable_ultrasonic_sensors: bool = False
    """启用超声波传感器 (4 个)"""

    enable_master_arm_data: bool = False
    """启用主臂数据（关节和夹爪关节状态）"""

    
    def get_enabled_sensors(self) -> List[str]:
        """Get all enabled sensors list"""
        enabled = []
        for field_name, field_value in self.__dict__.items():
            if field_name.startswith('enable_') and field_value:
                sensor_name = field_name.replace('enable_', '')
                enabled.append(sensor_name)
        return enabled
    
    def get_camera_names(self) -> List[str]:
        """Get enabled camera names list"""
        cameras = []
        if self.enable_head_rgb_stream:
            cameras.append('head_rgb_stream')
        if self.enable_head_depth_stream:
            cameras.append('head_depth_stream')
        if self.enable_left_arm_rgb_stream:
            cameras.append('left_arm_rgb_stream')
        if self.enable_right_arm_rgb_stream:
            cameras.append('right_arm_rgb_stream')
        return cameras

    @staticmethod
    def arm_gripper_cameras() -> "CollectionConfig":
        """Preset: record left/right arm joints, gripper info, and RGB streams from head/left_arm/right_arm cameras"""
        return CollectionConfig(
            slave_joint_names=[
                'left_arm_joint_states',
                'right_arm_joint_states',
                'left_gripper_joint_states',
                'right_gripper_joint_states',
            ],
            enable_left_gripper_position=True,
            enable_right_gripper_position=True,
            enable_head_rgb_stream=True,
            enable_left_arm_rgb_stream=True,
            enable_right_arm_rgb_stream=True,
        )

