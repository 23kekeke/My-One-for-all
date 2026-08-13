# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Record a dataset from XRobot (x2robot SDK) for imitation learning.

Example usage:

```shell
lerobot-record-x-robot \\
    --robot.type=x_robot \\
    --robot.server="192.168.36.116:50051" \\
    --robot.enable_head_camera=true \\
    --robot.enable_left_arm_camera=true \\
    --robot.enable_right_arm_camera=true \\
    --dataset.repo_id=my_user/xrobot_dataset \\
    --dataset.num_episodes=10 \\
    --dataset.single_task="Pick and place" \\
    --dataset.fps=30 \\
    --display_data=true

python -m lerobot.scripts.lerobot_record_x_robot \
    --robot.type=x_robot \
    --robot.server="192.168.36.116:50051" \
    --robot.enable_lift=true \
    --robot.head_camera.width=1280 \
    --robot.head_camera.height=720 \
    --dataset.repo_id=my_name/xrobot_dataset \
    --dataset.root=/home/hpc/datasets/2026_test_0526_test \
    --dataset.num_episodes=100 \
    --dataset.single_task="Pick up the express package and put it on the tray." \
    --display_data=true \
    --resume=false \
    --robot.log_alignment_stats true
```
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import torch

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy

# from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    # bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    # openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.robots.x_robot.config_x_robot import XRobotConfig  # noqa: F401
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    # bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    # openarm_leader,
    # openarm_mini,
    reachy2_teleoperator,
    so_leader,
    # unitree_g1,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import (
    init_keyboard_listener,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

import numpy as np


@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 120
    reset_time_s: int | float = 2  # 重置时间
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 4
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # dataclass 自动生成的 __init__ 执行完成后，
        # 立刻调用 __post_init__，用于做参数校验或派生字段初始化
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        # if self.teleop is None and self.policy is None:
        #     raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # 遥操作动作的后处理流水线（在人类输入之后）
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # 最终发送给机器人的动作处理流水线（发送前的统一处理）
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # 机器人观测的处理流水线（从 robot.get_observation() 之后）
    dataset: LeRobotDataset | None = None,  # 可选：用于写入数据集
    teleop: Teleoperator | list[Teleoperator] | None = None,  # 遥操作设备（单个或多个）
    policy: PreTrainedPolicy | None = None,  # 可选：策略模型
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,  # 当前 episode 的最大控制时长（秒）
    single_task: str | None = None,  # 当前 episode 的语言任务描述
    display_data: bool = False,  # 是否实时显示观测与动作
):

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    if policy is not None:
        policy.reset()

    threshold_update_timeout_s = 20  # 关节角信息不更新的时间超过阈值则认为是异常
    zero_joints_timeout_s = 10  # 关节角数据持续为0的时间超过阈值则认为是异常

    last_update_t = time.perf_counter()
    last_observation = {}
    zero_joints_start_t = None
    all_joints_zero = False

    # --------------------------
    # episode 时间控制
    # --------------------------
    timestamp = 0
    start_episode_t = time.perf_counter()
    frame_count = 0

    # 主控制循环（每一帧 / timestep）
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # 允许外部事件中断循环（如 Ctrl+C、UI 停止）
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 1. 获取机器人原始观测
        t0 = time.perf_counter()
        obs = robot.get_observation()
        
        # print("关节数据:")
        # for k in robot.motors.keys():
        #     print(f"  {k}: {obs.get(k, None)}")
        
        t_get_obs = time.perf_counter() - t0

        # Monitor joint state health for x_robot
        if robot.name == "x_robot":
            motor_keys = list(robot.motors.keys())
            all_zeros = True
            for obs_key in motor_keys:
                val = obs.get(obs_key, 0.0)
                if isinstance(val, (float, int)):
                    if val != last_observation.get(obs_key, None):
                        last_update_t = time.perf_counter()
                        last_observation[obs_key] = val
                    if abs(float(val)) > 1e-6:
                        all_zeros = False

            dt_update_s = time.perf_counter() - last_update_t
            if dt_update_s > threshold_update_timeout_s:
                logging.warning(f"Joint state data has not changed for more than {threshold_update_timeout_s}s")

            if all_zeros:
                if not all_joints_zero:
                    zero_joints_start_t = time.perf_counter()
                    all_joints_zero = True
                elif time.perf_counter() - zero_joints_start_t > zero_joints_timeout_s:
                    logging.warning(f"No valid joint angle data detected for more than {zero_joints_timeout_s}s")
                    zero_joints_start_t = time.perf_counter()
            else:
                all_joints_zero = False
                zero_joints_start_t = None

        # 对原始观测进行统一处理（默认是 IdentityProcessor）
        obs_processed = robot_observation_processor(obs)

        # 如果需要写数据集或使用策略，则构建标准 observation frame

        # print("填入数据")
        # print(dataset.features)
        # print("-----------------------")
        # print(obs_processed)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # 获取人类输入的原始动作
        action: dict[str, Any] = {}  # 动作
        t_get_action = time.perf_counter()
        sent_action = robot.get_action(action)
        # 加在这里
        # print("动作数据:")
        # for k, v in sent_action.items():
        #     print(f"  {k}: {v}")
        t_get_action = time.perf_counter() - t_get_action
        # act = teleop.get_action()

        # 对遥操作动作进行处理（坐标变换 / 平滑 / 限幅等）
        act_processed = teleop_action_processor((sent_action, obs))

        print("act_processed:")
        for k, v in act_processed.items():
            print(f"  {k}: {v}")
        # 5. 写入数据集
        if dataset is not None:
            t_write = time.perf_counter()
            action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
            print("action_frame:", action_frame)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
            t_write = time.perf_counter() - t_write
        else:
            t_write = 0
        # 6. 实时显示（调试 / 可视化）


        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed)

        # 7. FPS 控制与时间推进
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)
        frame_count += 1

        # Periodic state logging (every 300 frames = ~10s)
        if frame_count % 300 == 0:
            elapsed_so_far = time.perf_counter() - start_episode_t
            state_items = {k: round(float(v), 4) for k, v in obs_processed.items() if isinstance(v, (float, int, np.floating))}
            logging.info(
                f"Frame {frame_count} | fps={frame_count/elapsed_so_far:.1f} | state={state_items}"
            )

        timestamp = time.perf_counter() - start_episode_t

    # Log actual recording FPS
    elapsed = time.perf_counter() - start_episode_t
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    logging.info(
        f"Episode finished: {frame_count} frames in {elapsed:.2f}s, "
        f"actual fps={actual_fps:.1f}, target fps={fps}"
    )


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    # --------------------------
    # 1. 初始化日志与可视化
    # --------------------------
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 是否启用实时数据可视化（Rerun）
    if cfg.display_data:
        init_rerun(session_name="recording")

    # --------------------------
    # 2. 构建机器人与遥操作设备
    # --------------------------
    robot = make_robot_from_config(cfg.robot)

    # x_robot auto-detects robot model on connect; camera shapes come from config
    if robot.name == "x_robot":
        # Ensure camera configs have valid width/height for feature schema
        for cam_key in robot._camera_keys:
            shape = robot._get_camera_shape(cam_key)
            logging.info(f"x_robot camera '{cam_key}': shape={shape}")

    # 根据配置创建遥操作器（如果有）
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # --------------------------
    # 3. 构建默认处理流水线
    # --------------------------
    # - teleop_action_processor: 处理人类输入的动作
    # - robot_action_processor: 最终发送给机器人的动作
    # - robot_observation_processor: 处理机器人原始观测
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # --------------------------
    # 4. 构建数据集 feature schema
    # --------------------------
    # 将 action pipeline 和 observation pipeline 各自产生的 feature 进行合并
    print("------------------------------------------------")
    print(robot.action_features)
    print("------------------------------------------------")
    print(robot.observation_features)

    # 构建数据集特征
    dataset_features = combine_feature_dicts(
        # 动作相关 features
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        # 观测相关 features
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    print("数据集特征")
    print(dataset_features)

    dataset = None
    listener = None

    try:
        # --------------------------
        # 5. 创建或恢复数据集
        # --------------------------
        if cfg.resume:
            # 从已有数据集继续录制
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
            # 如果机器人有相机，启动图像写入器
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            # 校验数据集与机器人、fps、feature 的一致性
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # 创建一个新的数据集
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

        # --------------------------
        # 6. 加载预训练策略（如果有）
        # --------------------------
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            # 构建 policy 的前处理 / 后处理流水线
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    # 强制指定推理设备
                    "device_processor": {"device": cfg.policy.device},
                    # 观测 key 重命名
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        # --------------------------
        # 7. 连接机器人与遥操作设备
        # --------------------------
        robot.connect()
        if teleop is not None:
            teleop.connect()
        # 启动键盘监听器（停止 / 重录 / 退出）
        listener, events = init_keyboard_listener()
        # --------------------------
        # 8. 录制 episode 主循环
        # --------------------------
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                # Idle state: interactive terminals require key '3'; non-interactive runs auto-start.
                events["start_recording"] = listener is None
                events["exit_early"] = False

                log_say(
                    "Controls: 0=stop, 1=finish/save, 2=rerecord, 3=start recording",
                    cfg.play_sounds,
                )
                while not events["start_recording"] and not events["stop_recording"]:
                    # Ignore stale events while waiting for the next episode.
                    if events["exit_early"]:
                        events["exit_early"] = False
                    if events["rerecord_episode"]:
                        events["rerecord_episode"] = False
                    if events["start_recording"]:
                        events["start_recording"] = False
                    time.sleep(0.1)
                if events["stop_recording"]:
                    print("Stop recording")
                    break
                # 3 was pressed, reset flag and proceed
                events["start_recording"] = False
                log_say(
                    f"Recording episode {recorded_episodes + 1}/{cfg.dataset.num_episodes}",
                    cfg.play_sounds,
                )
                # Start gRPC streams for low-latency data fetching
                if robot.name == "x_robot":
                    robot.start_streams()
                # 正式录制一个 episode
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )
                # Stop streams to avoid keepalive pings during idle
                if robot.name == "x_robot":
                    robot.stop_streams()

                episode_size = dataset.writer.episode_buffer["size"]

                # --------------------------
                # 重录当前 episode
                # --------------------------
                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                if events["stop_recording"]:
                    if episode_size > 0:
                        dataset.save_episode()
                        recorded_episodes += 1
                    break

                if episode_size == 0:
                    logging.warning("Current episode is empty. Waiting for Space to start a new recording.")
                    continue

                # 保存当前 episode
                dataset.save_episode()
                recorded_episodes += 1
    finally:
        # --------------------------
        # 9. 清理与资源释放（保证执行）
        # --------------------------
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener:
            listener.stop()
        # --------------------------
        # 10. 上传数据集到 Hugging Face Hub
        # --------------------------
        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
