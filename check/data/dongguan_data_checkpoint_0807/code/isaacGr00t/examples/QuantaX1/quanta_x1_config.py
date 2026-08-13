# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Quanta X1 right-arm 16D LeRobot dataset (dataset/quanta_lerobot).
#
# Parquet vector layout (observation.state / action, 16D):
#   [0:9]   eef_9d          xyz(3) + rot6d(6) from right_arm_end_pose
#   [9:10]  gripper_position
#   [10:16] joint_position  6D right arm joints
#
# Action semantics in parquet: absolute next observed state (action[t] == state[t+1]).
#
# Language: parquet stores task_index (int); text lives in meta/tasks.jsonl and is
# resolved at load time via annotation.language.language_instruction.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# N1.7 default action chunk horizon
ACTION_HORIZON = 40

quanta_x1_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_camera",
            "right_arm_camera",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_9d",
            "gripper_position",
            "joint_position",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "eef_9d",
            "gripper_position",
            "joint_position",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.language.language_instruction",
        ],
    ),
}

register_modality_config(quanta_x1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
