# Pipeline7: state 32D / action 20D (eef-only relative) — GR00T modality config.
#
# State: same as pipeline5 (eef + gripper + joint ×2), delta [-7,-1,0]
# Action: eef RELATIVE + gripper ABSOLUTE only (no joint in action)

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

ACTION_REPS = [
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
    "ABSOLUTE",
]
EXPECTED_ACTION_REPS = ACTION_REPS

dongguan_eef_only_relative_config = {
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
            "right_eef_9d",
            "right_gripper_position",
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
                rep=_EEF_REL.rep,
                type=_EEF_REL.type,
                format=_EEF_REL.format,
                state_key="right_eef_9d",
            ),
            _GRIP_ABS,
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
    dongguan_eef_only_relative_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
