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
Replays the actions of an episode from a dataset on a robot.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0

适用与 x_robot
python -m lerobot.scripts.lerobot_replay_x_robot \
    --robot.type=x_robot \
    --robot.server="192.168.36.116:50051" \
    --dataset.repo_id=my_name/xrobot_dataset \
    --dataset.root=/home/hpc/datasets/2026_test_0526_test \
    --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.x_robot.config_x_robot import XRobotConfig  # noqa: F401
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


def _map_joint_names_to_motor_keys(action_dict: dict[str, float]) -> dict[str, float]:
    """将数据集 action 命名转为 send_action 认识的 motor key 格式。"""
    import re

    # 已知的非关节 key 映射（数据集命名 → motor key）
    KNOWN_MAP = {
        "left_gripper": "left_gripper.pos",
        "right_gripper": "right_gripper.pos",
        "follow_left_gripper": "left_gripper.pos",
        "follow_right_gripper": "right_gripper.pos",
        "height": "lift.pos",
    }

    mapped = {}
    for k, v in action_dict.items():
        if k in KNOWN_MAP:
            mapped[KNOWN_MAP[k]] = float(v)
        elif k.endswith(".pos"):
            mapped[k] = float(v)
        else:
            m = re.match(r"(.+?)_joint_?(\d+)?$", k)
            if m:
                prefix = m.group(1)
                idx = m.group(2)
                if idx is not None:
                    motor_key = f"{prefix}_{int(idx) - 1}.pos"
                    mapped[motor_key] = float(v)
    return mapped


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)

    robot.connect()
    if robot.name == "x_robot":
        robot.start_streams()

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        # joint_log_f = open(Path(f"joint_values_episode{cfg.dataset.episode}.txt"), "w")
        # joint_log_f.write("frame_idx\t" + "\t".join(dataset.features[ACTION]["names"]) + "\n")

        for idx in range(dataset.num_frames):
            start_episode_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            # joint_log_f.write(str(idx) + "\t" + "\t".join(str(float(action_array[i])) for i in range(len(action_array))) + "\n")

            robot_obs = robot.get_observation()

            processed_action = robot_action_processor((action, robot_obs))

            # 将数据集动作的 joint 命名转为 motor key 格式
            arm_action = _map_joint_names_to_motor_keys(action)

            # 对于其他电机 key（如 head_*.pos），保持当前观测值，
            # 避免 send_action 默认 0.0 导致意外运动。
            full_action = dict(arm_action)
            for k in robot.motors:
                if k not in full_action and k in robot_obs:
                    full_action[k] = float(robot_obs[k])

            print(full_action)

            _ = robot.send_action(full_action)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        # joint_log_f.close()
        logging.info(f"Joint values saved to joint_values_episode{cfg.dataset.episode}.txt")
        if robot.name == "x_robot":
            robot.stop_streams()
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
