# Dongguan bimanual 32D — RELATIVE action training config (pipeline5).
#
# Parquet semantics unchanged: action[t] == state[t+1]
# Train-time relative conversion via action_configs + use_relative_action=True.
#
# delta_indices [-7, -1, 0] @ 15fps ≈ 0.47s / 0.07s / current (3-frame history).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 40

OBSERVATION_DELTA_INDICES = [-7, -1, 0]
STATE_HISTORY_LENGTH = len(OBSERVATION_DELTA_INDICES)

_EEF_REL = ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.EEF,
    format=ActionFormat.XYZ_ROT6D,
)
_GRIP_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_JOINT_REL = ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

ACTION_REPS = [
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
]

dongguan_relative_config = {
    "video": ModalityConfig(
        delta_indices=list(OBSERVATION_DELTA_INDICES),
        modality_keys=[
            "head_camera",
            "left_arm_camera",
            "right_arm_camera",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=list(OBSERVATION_DELTA_INDICES),
        modality_keys=[
            "left_eef_9d",
            "left_gripper_position",
            "left_joint_position",
            "right_eef_9d",
            "right_gripper_position",
            "right_joint_position",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "left_eef_9d",
            "left_gripper_position",
            "left_joint_position",
            "right_eef_9d",
            "right_gripper_position",
            "right_joint_position",
        ],
        action_configs=[
            ActionConfig(
                rep=_EEF_REL.rep,
                type=_EEF_REL.type,
                format=_EEF_REL.format,
                state_key="left_eef_9d",
            ),
            _GRIP_ABS,
            ActionConfig(
                rep=_JOINT_REL.rep,
                type=_JOINT_REL.type,
                format=_JOINT_REL.format,
                state_key="left_joint_position",
            ),
            ActionConfig(
                rep=_EEF_REL.rep,
                type=_EEF_REL.type,
                format=_EEF_REL.format,
                state_key="right_eef_9d",
            ),
            _GRIP_ABS,
            ActionConfig(
                rep=_JOINT_REL.rep,
                type=_JOINT_REL.type,
                format=_JOINT_REL.format,
                state_key="right_joint_position",
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

register_modality_config(
    dongguan_relative_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
