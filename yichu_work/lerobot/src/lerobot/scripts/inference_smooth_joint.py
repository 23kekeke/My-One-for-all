import numpy as np
import torch
import time
from collections import deque
import requests
import threading
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

# 导入机器人模型
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
    # moving_dual_piper,
)
# 导入相机相关模块
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from dataclasses import asdict, dataclass
from lerobot.utils.robot_utils import busy_wait
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
# from lerobot.utils.utils import (
#     get_safe_torch_device,
#     init_logging,
#     log_say,
# )
# 导入策略模型
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
# from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs import parser
# 导入piper SDK
from piper_sdk import *

@dataclass
class InferenceConfig:
    """推理配置类，用于存储机器人配置和策略配置"""
    robot: RobotConfig
    # 是否使用策略控制机器人
    policy: PreTrainedConfig | None = None
    # 模型检查点路径
    ckpt_path: str = None
    # 任务名称
    task: str = None

    def __post_init__(self):
        """初始化后处理，从命令行参数获取预训练模型路径"""
        # 从命令行参数获取策略路径
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            # 从预训练路径加载配置
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """返回路径字段列表，使解析器能够通过--policy.path=local/dir加载配置"""
        return ["policy"]


def enable_fun(piper: C_PiperInterface):
    '''
    使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序
    '''
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        
    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)


def generate_smooth_curve(start_val, end_val, start_slope, end_slope, transition_steps, smooth_factor=1.0):
    """
    三次多项式插值生成平滑曲线
    :param start_val: 起点值（list/np.array）
    :param end_val: 终点值（list/np.array）
    :param start_slope: 起点前斜率（np.array）
    :param end_slope: 终点后斜率（np.array）
    :param transition_steps: 过渡步数
    :param smooth_factor: 平缓系数（0~1，越小曲线越平缓）
    :return: 平滑后的过渡曲线（np.array），shape=(transition_steps+1, dim)
    """
    # 统一转换为numpy数组，兼容list输入
    start_val = np.array(start_val, dtype=np.float32)
    end_val = np.array(end_val, dtype=np.float32)
    start_slope = np.array(start_slope, dtype=np.float32)
    end_slope = np.array(end_slope, dtype=np.float32)

    if start_val.ndim == 0:
        start_val = start_val.reshape(1)
        end_val = end_val.reshape(1)
        start_slope = start_slope.reshape(1)
        end_slope = end_slope.reshape(1)

    dim = start_val.shape[0]
    smoothed_batch = np.zeros((transition_steps + 1, dim), dtype=np.float32)

    for step_idx in range(transition_steps + 1):
        # 归一化
        t = step_idx / transition_steps
        t2 = t * t
        t3 = t2 * t

        # 基础线性过渡
        linear_part = start_val + (end_val - start_val) * t

        # 三次多项式修正项
        a1 = start_slope
        a2 = 3 * (end_val - start_val) - 2 * a1 - end_slope
        a3 = 2 * (start_val - end_val) + a1 + end_slope
        cubic_correction = (a1 * t + a2 * t2 + a3 * t3) - (end_val - start_val) * t

        smoothed_batch[step_idx] = linear_part + cubic_correction * smooth_factor

    if dim == 1:
        return smoothed_batch.squeeze(axis=1)
    return smoothed_batch


class RobotInferenceSystem:
    """机器人推理系统，封装线程化的采集+推理、动作执行逻辑"""
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.robot = None
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.device = "cuda"
        self.factor = 57324.840764  # 关节角度到控制信号的缩放因子 1000 * 180 / 3.14
        
        self.running = False                        # 线程控制标志
        self.infer_collect_thread = None            # 采集+推理合并线程
        self.execute_thread = None                  # 动作执行线程
        self.last_executed_action = None            # 上次最新执行的动作
        self.second_to_last_executed_action = None  # 倒数第二次执行的动作

        self.SMOOTH_WINDOW = 11             # 批次内平滑窗口大小
        self.INFERENCE_PERIOD = 0.8         # 采集+推理周期，秒
        self.EXECUTION_PERIOD = 0.05       # 执行周期，秒
        
        # ROS相关（仅移动双臂机器人）
        self.cmd_vel_pub = None
        
        # 关节限位
        self.joint_limits = [(-3, 3)] * 6
        self.joint_limits[0] = (-2.687, 2.687)     # 关节0限位
        self.joint_limits[1] = (0.0, 3.403)        # 关节1限位
        self.joint_limits[2] = (-3.0541012, 0.0)   # 关节2限位
        self.joint_limits[3] = (-1.5499, 1.5499)   # 关节3限位
        self.joint_limits[4] = (-1.22, 1.22)       # 关节4限位
        self.joint_limits[5] = (-1.7452, 1.7452)   # 关节5限位

    def init_robot(self):
        """初始化机器人并完成使能"""
        # 创建机器人实例并连接
        self.robot = make_robot_from_config(self.cfg.robot)
        if not self.robot.is_connected:
            self.robot.connect()
        
        # 如果是移动双Piper机器人，初始化ROS节点
        # if cfg.robot.type == "moving_dual_piper":
        #     import rospy
        #     from geometry_msgs.msg import Twist
        #     rospy.init_node("pub_cmd_vel", anonymous=True)
        #     cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        #     print(f"using moving_dual_piper robot, ROS node 'pub_cmd_vel' initialized.")
        
        # 初始化策略模型
        self._init_policy()
        
        # 机器人使能
        self._enable_robot()
        
        # 等待使能完成
        time.sleep(2.0)

    def _init_policy(self):
        """初始化策略模型"""
        if self.cfg.task is None:
            raise ValueError("You need to provide a task name.")
        
        ckpt_path = self.cfg.ckpt_path
        print(f"ckpt_path : {ckpt_path}")
        
        # 根据策略类型创建模型
        policy_type = self.cfg.policy.type
        if policy_type == "act":
            self.policy = ACTPolicy.from_pretrained(ckpt_path)
        elif policy_type == "diffusion":
            self.policy = DiffusionPolicy.from_pretrained(ckpt_path)
        elif policy_type == "smolvla":
            self.policy = SmolVLAPolicy.from_pretrained(ckpt_path)
        elif policy_type == "groot":
            self.policy = GrootPolicy.from_pretrained(ckpt_path)
        elif policy_type == "xvla":
            self.policy = XVLAPolicy.from_pretrained(ckpt_path)
        elif policy_type == "pi05":
            self.policy = PI05Policy.from_pretrained(ckpt_path)
        else:
            raise ValueError("You need to provide a valid policy between act/diffusion/smolvla/xvla.")
        
        self.position_queue = deque(maxlen=self.policy.config.n_action_steps)  # 位置队列
        self.raw_action_queue = deque(maxlen=self.policy.config.n_action_steps)  # 原始动作队列

        self.position_queue_lock = threading.Lock()

        # 初始化预处理/后处理函数
        self.preprocess, self.postprocess = make_pre_post_processors(self.policy.config, ckpt_path)
        
        # 模型准备
        self.policy.eval()
        self.policy.to(self.device)
        self.policy.reset()

    def _enable_robot(self):
        """根据机器人类型执行使能操作"""
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            # 单臂机器人
            self.robot.piper.EnableArm(7)
            enable_fun(piper=self.robot.piper)
            self.robot.piper.MotionCtrl_2(0x01, 0x01, 10, 0x00)
            self.robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
        elif robot_type in ["dual_piper", "moving_dual_piper"]:
            # 双臂机器人
            # 左臂使能
            self.robot.piper_left.EnableArm(7)
            enable_fun(piper=self.robot.piper_left)
            self.robot.piper_left.MotionCtrl_2(0x01, 0x01, 60, 0x00)
            self.robot.piper_left.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
            # 右臂使能
            self.robot.piper_right.EnableArm(7)
            enable_fun(piper=self.robot.piper_right)
            self.robot.piper_right.MotionCtrl_2(0x01, 0x01, 60, 0x00)
            self.robot.piper_right.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

    def _collect_and_infer(self):
        """采集+推理线程函数"""
        while self.running:
            start_t = time.perf_counter()
            try:
                # 采集观测数据
                observation = self.robot.get_observation()
                observation_frame = {}

                # 记录观测后，推理前的队列大小
                with self.position_queue_lock:
                    action_queue_len_pre = len(self.position_queue)

                state_values = []
                state_values_epos = []

                # 处理状态和图像数据
                for key, value in observation.items():
                    if key.endswith('.pos'):
                        state_values.append(np.float32(value))
                    if key.endswith('.epos'):
                        state_values_epos.append(np.float32(value))
                    elif isinstance(value, np.ndarray) and value.ndim == 3:
                        observation_frame[f'observation.images.{key}'] = value

                # 组装观测帧
                observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)
                
                # 预处理：图像归一化、增加batch维度、转tensor
                for name in observation_frame:
                    if "image" in name:
                        obs = observation_frame[name].astype(np.float32) / 255.0  # 归一化
                        obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
                    else:
                        obs = observation_frame[name]

                    obs = np.expand_dims(obs, axis=0)
                    observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
                observation_frame["task"] = [self.cfg.task]

                # 策略预处理
                observation_frame = self.preprocess(observation_frame)

                # 策略推理动作
                with torch.inference_mode():
                    actions = self.policy.predict_action_chunk(observation_frame)[:, : self.policy.config.n_action_steps]
                
                # 动作平滑
                actions_np = actions.squeeze(0).cpu().detach().numpy()
                smoothed_batch = np.copy(actions_np)
                
                # 批次内平滑
                pad = self.SMOOTH_WINDOW // 2
                for dim in range(actions.shape[-1]):
                    window = np.ones(self.SMOOTH_WINDOW) / self.SMOOTH_WINDOW
                    col = smoothed_batch[:, dim]
                    col_padded = np.pad(col, pad_width=(pad, pad), mode='reflect')
                    smoothed_col = np.convolve(col_padded, window, mode='valid')
                    smoothed_batch[:, dim] = smoothed_col

                # 剔除新批次动作中前边已过时的动作
                with self.position_queue_lock:
                    executed_steps = action_queue_len_pre - len(self.position_queue)
                if executed_steps > 0:
                    if executed_steps < len(smoothed_batch):
                        smoothed_batch = smoothed_batch[executed_steps:]
                    else:
                        smoothed_batch = smoothed_batch[-1:]

                # 批次间平滑
                TRANSITION_STEPS = 12
                # if self.last_executed_action is not None and smoothed_batch.shape[0] > TRANSITION_STEPS:
                #     with self.position_queue_lock:
                #         start_point = self.last_executed_action
                #     end_point = smoothed_batch[TRANSITION_STEPS]
                #     for step_idx in range(TRANSITION_STEPS):
                #         weight = step_idx / TRANSITION_STEPS
                #         smoothed_batch[step_idx] = start_point + weight * (end_point - start_point)
                # if self.last_executed_action is not None and smoothed_batch.shape[0] > TRANSITION_STEPS:
                #     with self.position_queue_lock:
                #         start_point = self.last_executed_action
                #     for step_idx in range(TRANSITION_STEPS):
                #         weight = (step_idx / TRANSITION_STEPS) ** 2
                #         smoothed_batch[step_idx] = start_point + weight * (smoothed_batch[step_idx] - start_point)
                if (self.last_executed_action is not None and self.second_to_last_executed_action is not None and
                        smoothed_batch.shape[0] > TRANSITION_STEPS):
                    with self.position_queue_lock:
                        start_point = self.last_executed_action
                    end_point = smoothed_batch[TRANSITION_STEPS]
                    start_slope = start_point - self.second_to_last_executed_action
                    if len(smoothed_batch) > TRANSITION_STEPS + 1:
                        end_slope = smoothed_batch[TRANSITION_STEPS + 1] - end_point
                    else:
                        end_slope = np.zeros_like(end_point)

                    smooth_curve = generate_smooth_curve(start_point, end_point, start_slope, end_slope, TRANSITION_STEPS)
                    if len(smooth_curve) <= len(smoothed_batch):
                        smoothed_batch[:len(smooth_curve)] = smooth_curve
                    else:
                        smoothed_batch = smooth_curve[:len(smoothed_batch)]

                # 将平滑后的动作批量转换为可执行的position列表
                position_list = []
                raw_action_list = []
                for action_step in smoothed_batch:
                    raw_action_list.append(action_step)

                    # 转换为torch tensor并后处理
                    action_tensor = torch.from_numpy(action_step).unsqueeze(0).to(self.device)
                    action_post = self.postprocess(action_tensor)
                    
                    # 转换为numpy数组
                    if self.cfg.policy.type == "xvla":
                        numpy_action = action_post.squeeze(0).cpu().to(torch.float32).numpy()
                    else:
                        numpy_action = action_post.squeeze(0).cpu().numpy()
                    
                    # 转换为列表并添加
                    position_list.append(numpy_action.tolist())

                # 放入动作队列
                with self.position_queue_lock:
                    self.position_queue.clear()
                    self.raw_action_queue.clear()
                    for pos, raw in zip(position_list, raw_action_list):
                        self.position_queue.append(pos)
                        self.raw_action_queue.append(raw)
                
                # 控制采集+推理频率
                dt = time.perf_counter() - start_t
                print(f"inference over. cost time:{dt}")
                wait_time = self.INFERENCE_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"采集+推理线程异常: {e}")
                time.sleep(0.001)

    def _execute_action(self):
        """动作执行线程函数（独立线程，保证控制实时性）"""
        last_queue_len = 1000
        is_new_batch = False
        while self.running:
            start_t = time.perf_counter()
            position = None
            raw_action = None
            try:
                # 获取动作数据，阻塞等待，直到有数据
                while self.running:
                    with self.position_queue_lock:
                        if self.position_queue:
                            is_new_batch = False
                            if last_queue_len < len(self.position_queue):
                                is_new_batch = True
                            last_queue_len = len(self.position_queue)
                            position = self.position_queue.popleft()
                            raw_action = self.raw_action_queue.popleft()
                            break
                    time.sleep(0.001)
                if not self.running:
                    break

                self.second_to_last_executed_action = self.last_executed_action
                self.last_executed_action = raw_action

                # 执行动作
                print(f"{'new ' if is_new_batch else ''}{position}")
                self._execute_robot_action(position)

                # 控制执行频率
                dt = time.perf_counter() - start_t
                # print(f"execute end. cost time:{dt}")
                wait_time = self.EXECUTION_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"动作执行线程异常: {e}")
                time.sleep(0.001)

    def _execute_robot_action(self, position):
        """根据机器人类型执行具体的动作控制"""
        robot_type = self.cfg.robot.type
        clamp = lambda v, min_v, max_v: max(min(v, max_v), min_v)  # 限位函数
        
        if robot_type == "single_piper":
            # 单臂控制
            joint_0 = round(clamp(position[0], *self.joint_limits[0]) * self.factor)
            joint_1 = round(clamp(position[1], *self.joint_limits[1]) * self.factor)
            joint_2 = round(clamp(position[2], *self.joint_limits[2]) * self.factor)
            joint_3 = round(clamp(position[3], *self.joint_limits[3]) * self.factor)
            joint_4 = round(clamp(position[4], *self.joint_limits[4]) * self.factor)
            joint_5 = round(clamp(position[5], *self.joint_limits[5]) * self.factor)
            joint_6 = round(position[6] * 70 * 1000)
            
            self.robot.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            self.robot.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        
        elif robot_type == "dual_piper":
            # 双臂控制
            # 左臂
            left_joint_0 = round(clamp(position[0], *self.joint_limits[0]) * self.factor)
            left_joint_1 = round(clamp(position[1], *self.joint_limits[1]) * self.factor)
            left_joint_2 = round(clamp(position[2], *self.joint_limits[2]) * self.factor)
            left_joint_3 = round(clamp(position[3], *self.joint_limits[3]) * self.factor)
            left_joint_4 = round(clamp(position[4], *self.joint_limits[4]) * self.factor)
            left_joint_5 = round(clamp(position[5], *self.joint_limits[5]) * self.factor)
            left_joint_6 = round(position[6] * 70 * 1000)
            
            # 右臂
            right_joint_0 = round(clamp(position[7], *self.joint_limits[0]) * self.factor)
            right_joint_1 = round(clamp(position[8], *self.joint_limits[1]) * self.factor)
            right_joint_2 = round(clamp(position[9], *self.joint_limits[2]) * self.factor)
            right_joint_3 = round(clamp(position[10], *self.joint_limits[3]) * self.factor)
            right_joint_4 = round(clamp(position[11], *self.joint_limits[4]) * self.factor)
            right_joint_5 = round(clamp(position[12], *self.joint_limits[5]) * self.factor)
            right_joint_6 = round(position[13] * 70 * 1000)
            
            self.robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
            self.robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
            self.robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
            self.robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        
        elif robot_type == "moving_dual_piper" and not rospy.is_shutdown():
            # 移动双臂机器人控制 - 底盘控制
            # 底盘控制
            twist_msg = Twist()
            twist_msg.linear.x = position[0]
            twist_msg.angular.z = position[1]
            self.cmd_vel_pub.publish(twist_msg)
            
            # 左臂
            left_joint_0 = round(clamp(position[2], *self.joint_limits[0]) * self.factor)
            left_joint_1 = round(clamp(position[3], *self.joint_limits[1]) * self.factor)
            left_joint_2 = round(clamp(position[4], *self.joint_limits[2]) * self.factor)
            left_joint_3 = round(clamp(position[5], *self.joint_limits[3]) * self.factor)
            left_joint_4 = round(clamp(position[6], *self.joint_limits[4]) * self.factor)
            left_joint_5 = round(clamp(position[7], *self.joint_limits[5]) * self.factor)
            left_joint_6 = round(position[8] * 70 * 1000)
            
            # 右臂
            right_joint_0 = round(clamp(position[9], *self.joint_limits[0]) * self.factor)
            right_joint_1 = round(clamp(position[10], *self.joint_limits[1]) * self.factor)
            right_joint_2 = round(clamp(position[11], *self.joint_limits[2]) * self.factor)
            right_joint_3 = round(clamp(position[12], *self.joint_limits[3]) * self.factor)
            right_joint_4 = round(clamp(position[13], *self.joint_limits[4]) * self.factor)
            right_joint_5 = round(clamp(position[14], *self.joint_limits[5]) * self.factor)
            right_joint_6 = round(position[15] * 70 * 1000)
            
            self.robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
            self.robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
            self.robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
            self.robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type for action execution: {robot_type}")

    def start(self):
        if self.running:
            return
        
        # 初始化机器人和策略
        self.init_robot()
        
        # 设置运行标志
        self.running = True
        
        # 启动线程（采集+推理 合并线程、动作执行线程）
        self.infer_collect_thread = threading.Thread(target=self._collect_and_infer, daemon=True)
        self.execute_thread = threading.Thread(target=self._execute_action, daemon=True)
        
        self.infer_collect_thread.start()
        self.execute_thread.start()
        
        print("所有线程已启动，开始机器人控制循环...")

    def stop(self):
        """停止所有线程并清理资源"""
        self.running = False
        
        # 等待线程结束
        if self.infer_collect_thread:
            self.infer_collect_thread.join(timeout=1.0)
        if self.execute_thread:
            self.execute_thread.join(timeout=1.0)
        
        print("所有线程已停止")


@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """推理入口函数"""
    # 创建推理系统实例
    inference_system = RobotInferenceSystem(cfg)
    
    try:
        # 启动线程化推理
        inference_system.start()
        
        # 主进程空闲运行
        inference_time_s = 3600  # 推理总时间, 1小时
        start_time = time.time()
        while time.time() - start_time < inference_time_s:
            if not inference_system.running:
                break
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        print("\n接收到停止信号，正在关闭系统...")
    finally:
        # 停止所有线程
        inference_system.stop()
        print("推理系统已正常关闭")


if __name__ == "__main__":
    inference()
