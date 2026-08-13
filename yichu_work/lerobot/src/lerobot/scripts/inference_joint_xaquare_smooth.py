import logging
import time
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.x_robot.config_x_robot import XRobotConfig  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

@dataclass
class InferenceConfig:
    """推理配置类，用于存储机器人配置和策略配置。"""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    ckpt_path: str | None = None
    task: str | None = None

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def build_observation_frame(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    observation_frame: dict[str, np.ndarray] = {}
    state_values: list[np.float32] = []

    for key, value in observation.items():
        if key.endswith(".pos"):
            state_values.append(np.float32(value))
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            observation_frame[f"observation.images.{key}"] = value

    if not state_values:
        raise ValueError("当前观测中没有找到任何 `.pos` 状态，无法构造 `observation.state`。")

    observation_frame["observation.state"] = np.array(state_values, dtype=np.float32)
    return observation_frame


def generate_smooth_curve(start_val, end_val, start_slope, end_slope, transition_steps, smooth_factor=0.5):
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
        t = step_idx / transition_steps
        t2 = t * t
        t3 = t2 * t

        linear_part = start_val + (end_val - start_val) * t

        a1 = start_slope
        a2 = 3 * (end_val - start_val) - 2 * a1 - end_slope
        a3 = 2 * (start_val - end_val) + a1 + end_slope
        cubic_correction = (a1 * t + a2 * t2 + a3 * t3) - (end_val - start_val) * t

        smoothed_batch[step_idx] = linear_part + cubic_correction * smooth_factor

    if dim == 1:
        return smoothed_batch.squeeze(axis=1)
    return smoothed_batch


@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """主要的推理函数，执行机器人控制循环。"""

    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.policy is None:
        raise ValueError("You need to provide a policy configuration, for example `--policy.path=...`.")
    if cfg.task is None:
        raise ValueError("You need to provide a task name.")

    ckpt_path = cfg.ckpt_path or (
        str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path is not None else None
    )
    if ckpt_path is None:
        raise ValueError("You need to provide `--ckpt_path=...` or `--policy.path=...`.")

    inference_time_s = 10000
    fps = 30
    smooth_window = 16
    transition_steps = 18

    last_executed_action = None
    second_to_last_executed_action = None

    robot = make_robot_from_config(cfg.robot)

    try:
        if not robot.is_connected:
            robot.connect()

        robot.start_streams()

        policy_cls = get_policy_class(cfg.policy.type)
        policy = policy_cls.from_pretrained(ckpt_path, config=cfg.policy)
        device = get_safe_torch_device(policy.config.device)
        policy.to(device)
        policy.eval()
        policy.reset()

        device_override = {"device_processor": {"device": "cpu"}} if not torch.cuda.is_available() else {}
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            ckpt_path,
            preprocessor_overrides=device_override,
            postprocessor_overrides=device_override,
        )
        preprocess.reset()
        postprocess.reset()

        use_amp = policy.config.use_amp

        start_time = time.time()
        while time.time() - start_time < inference_time_s:
            observation = robot.get_observation()
            observation_frame = build_observation_frame(observation)

            obs_for_inference = copy(observation_frame)
            obs_for_inference = prepare_observation_for_inference(
                obs_for_inference, device, cfg.task, robot.robot_type
            )

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type)
                if device.type == "cuda" and use_amp
                else nullcontext(),
            ):
                obs_for_inference = preprocess(obs_for_inference)
                actions = policy.predict_action_chunk(obs_for_inference)[:, : policy.config.n_action_steps]

            actions_np = actions.squeeze(0).cpu().detach().numpy()
            smoothed_batch = np.copy(actions_np)

            pad = smooth_window // 2

            # 再不行，这里改为两次循环
            for dim in range(smoothed_batch.shape[1]):
                window = np.ones(smooth_window) / smooth_window
                col = smoothed_batch[:, dim]
                col_padded = np.pad(col, pad_width=(pad, pad), mode="reflect")
                smoothed_col = np.convolve(col_padded, window, mode="same")[pad:-pad]
                smoothed_batch[:, dim] = smoothed_col

            if (
                last_executed_action is not None
                and second_to_last_executed_action is not None
                and smoothed_batch.shape[0] > transition_steps
            ):
                start_point = last_executed_action
                end_point = smoothed_batch[transition_steps]
                start_slope = start_point - second_to_last_executed_action
                if len(smoothed_batch) > transition_steps + 1:
                    end_slope = smoothed_batch[transition_steps + 1] - end_point
                else:
                    end_slope = np.zeros_like(end_point)

                smooth_curve = generate_smooth_curve(
                    start_point, end_point, start_slope, end_slope, transition_steps
                )
                if len(smooth_curve) <= len(smoothed_batch):
                    smoothed_batch[: len(smooth_curve)] = smooth_curve
                else:
                    smoothed_batch = smooth_curve[: len(smoothed_batch)]

            print(f"Inference done, executing {len(smoothed_batch)} smoothed action steps.")

            for action_step in smoothed_batch:
                if time.time() - start_time >= inference_time_s:
                    break

                loop_start = time.perf_counter()

                second_to_last_executed_action = last_executed_action
                last_executed_action = np.copy(action_step)

                action_tensor = torch.from_numpy(action_step).unsqueeze(0).to(device)
                action_post = postprocess(action_tensor)
                numpy_action = action_post.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()

                motor_keys = list(robot.motors.keys())
                action_dict = {}
                for i, key in enumerate(motor_keys):
                    if i < len(numpy_action):
                        action_dict[key] = float(numpy_action[i])

                print(action_dict)
                robot.send_action(action_dict)

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1 / fps - dt_s, 0.0))
    finally:
        if robot.is_connected:
            robot.stop_streams()
            robot.disconnect()


def main():
    register_third_party_plugins()
    inference()


if __name__ == "__main__":
    main()
