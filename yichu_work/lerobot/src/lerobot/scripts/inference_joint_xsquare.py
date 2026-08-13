import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.x_robot.config_x_robot import XRobotConfig  # noqa: F401
from lerobot.common.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

try:
    import rerun as rr
except ImportError:
    rr = None

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
    fps = 150

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

        if rr is not None:
            rr.init("robot_inference")
            rr.spawn()

        for step_idx in range(inference_time_s * fps):
            start_loop_t = time.perf_counter()

            observation = robot.get_observation()
            observation_frame = build_observation_frame(observation)

            action = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocess,
                postprocessor=postprocess,
                use_amp=policy.config.use_amp,
                task=cfg.task,
                robot_type=robot.robot_type,
            )

            print("action shape:", action.shape)

            infer_time = time.perf_counter() - start_loop_t
            # print("action", action)
            print("infer_time", infer_time)

            motor_keys = list(robot.motors.keys())
            action_dict = {}
            numpy_action = action.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()
            for i, key in enumerate(motor_keys):
                if i < len(numpy_action):
                    action_dict[key] = float(numpy_action[i])

            # print(action_dict)
            robot.send_action(action_dict)

            print("action shape:", action.shape)

            if rr is not None:
                rr.set_time("step", sequence=step_idx)
                for key, value in action_dict.items():
                    rr.log(f"action/{key}", rr.Scalars(value))
                rr.log("action/bar", rr.BarChart(list(action_dict.values())))
                

            dt_s = time.perf_counter() - start_loop_t
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
