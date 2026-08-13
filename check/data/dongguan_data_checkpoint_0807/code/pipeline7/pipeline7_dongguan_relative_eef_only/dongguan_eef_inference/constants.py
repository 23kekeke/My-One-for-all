"""Train/infer contract for Dongguan pipeline7 eef-only relative GR00T.

State 32D (eef+gripper+joint ×2); action 20D (eef+gripper ×2, no joints).
action_reps: REL,ABS,REL,ABS
"""

from __future__ import annotations

import os
from pathlib import Path


def _detect_deploy_root() -> Path | None:
    explicit = os.environ.get("DONGGUAN_DEPLOY_ROOT")
    if explicit:
        root = Path(explicit)
        if (root / "models").is_dir() and (root / "checkpoints").is_dir():
            return root.resolve()
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "models").is_dir() and (ancestor / "checkpoints").is_dir():
            if ancestor.name == "deploy_dgx" or (ancestor / "per_task_home.json").is_file():
                return ancestor.resolve()
            if (ancestor / "code" / "pipeline7").is_dir():
                return ancestor.resolve()
            if (ancestor / "code" / "pipeline5_dongguan_relative").is_dir():
                return ancestor.resolve()
    return None


_DEPLOY_ROOT = _detect_deploy_root()

PIPELINE7 = Path(__file__).resolve().parents[1]

if _DEPLOY_ROOT is not None:
    DEPLOY_ROOT = _DEPLOY_ROOT
    CODE_ROOT = DEPLOY_ROOT / "code"
    GR00T_REPO = CODE_ROOT / "isaacGr00t"
    PIPELINE2 = CODE_ROOT / "pipeline2"
    PIPELINE3 = CODE_ROOT / "pipeline3_biman"
    ROOT = CODE_ROOT
    BASE_MODEL = DEPLOY_ROOT / "models/GR00T-N1.7-3B"
    COSMOS_MODEL = DEPLOY_ROOT / "models/Cosmos-Reason2-2B"
    CHECKPOINT_ROOT = DEPLOY_ROOT
    DEFAULT_CHECKPOINT = DEPLOY_ROOT / "checkpoints/multi345_end_only/checkpoint-15000"
    DEFAULT_TRAIN_OUTPUT = DEPLOY_ROOT / "checkpoints/multi345_end_only"
    MULTI_DATASET = DEPLOY_ROOT / "lerobot_p7/multi_345"
    if not MULTI_DATASET.is_dir():
        MULTI_DATASET = DEPLOY_ROOT / "lerobot/multi_345"
    INFERENCE_TMP = DEPLOY_ROOT / "tmp/dongguan_eef_inference"
    DEFAULT_PER_TASK_HOME = PIPELINE7 / "deploy" / "per_task_home.json"
    candidate = DEPLOY_ROOT / "per_task_home.json"
    if not DEFAULT_PER_TASK_HOME.is_file() and candidate.is_file():
        DEFAULT_PER_TASK_HOME = candidate
else:
    DEPLOY_ROOT = None
    ROOT = Path(__file__).resolve().parents[3]
    GR00T_REPO = ROOT / "isaacGr00t"
    PIPELINE2 = ROOT / "pipeline2"
    PIPELINE3 = ROOT / "pipeline3_biman"
    BASE_MODEL = ROOT / "GR00T-N1.7-3B"
    COSMOS_MODEL = ROOT / "Cosmos-Reason2-2B"
    CHECKPOINT_ROOT = Path("/data/dongguan_data_checkpoint_0807")
    DEFAULT_TRAIN_OUTPUT = CHECKPOINT_ROOT / "checkpoints/multi345_end_only"
    DEFAULT_CHECKPOINT = DEFAULT_TRAIN_OUTPUT / "checkpoint-15000"
    MULTI_DATASET = CHECKPOINT_ROOT / "lerobot_p7/multi_345"
    INFERENCE_TMP = PIPELINE7 / "tmp/dongguan_eef_inference"
    DEFAULT_PER_TASK_HOME = PIPELINE7 / "deploy" / "per_task_home.json"

LIVE_RUNS_TMP = INFERENCE_TMP / "live_runs"
HOME_PREPOSITION_DEFAULT_MAX_JOINT_DELTA_RAD = 0.10
HOME_PREPOSITION_DEFAULT_TOLERANCE_RAD = 0.05
HOME_PREPOSITION_DEFAULT_TOLERANCE_M = 0.025
HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD = 0.20
HOME_PREPOSITION_DEFAULT_MAX_STEPS = 1
HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC = 1.0
HOME_PREPOSITION_DEFAULT_ARM_SETTLE_SEC = 1.0
HOME_PREPOSITION_DEFAULT_INTERPOLATE_HZ = 10.0
HOME_PREPOSITION_DEFAULT_MAX_LINEAR_SPEED_M_S = 0.015
HOME_PREPOSITION_DEFAULT_MIN_DURATION_SEC = 5.0
HOME_PREPOSITION_DEFAULT_MAX_STEP_M = 0.008
OPEN_LOOP_TMP = INFERENCE_TMP / "open_loop"
GR00T_PYTHON = GR00T_REPO / ".venv/bin/python"

EMBODIMENT_TAG_VALUE = "new_embodiment"
ACTION_HORIZON = 40
NUM_INFERENCE_TIMESTEPS = 4
DEFAULT_EXECUTION_HORIZON = 8
LIVE_DEFAULT_EXECUTION_HORIZON = 1
STATE_DIM = 32
ACTION_DIM = 20
TRAIN_FPS = 15.0

# Must match dongguan_eef_only_relative_config.py
OBSERVATION_DELTA_INDICES = [-7, -1, 0]
STATE_HISTORY_LENGTH = len(OBSERVATION_DELTA_INDICES)
TEMPORAL_SPAN_SEC = abs(OBSERVATION_DELTA_INDICES[0]) / TRAIN_FPS
TEMPORAL_WARMUP_CYCLES = abs(OBSERVATION_DELTA_INDICES[0])

LANGUAGE_KEY = "annotation.language.language_instruction"
VIDEO_KEYS = ["head_camera", "left_arm_camera", "right_arm_camera"]
STATE_KEYS = [
    "left_eef_9d",
    "left_gripper_position",
    "left_joint_position",
    "right_eef_9d",
    "right_gripper_position",
    "right_joint_position",
]
ACTION_KEYS = [
    "left_eef_9d",
    "left_gripper_position",
    "right_eef_9d",
    "right_gripper_position",
]

EXPECTED_ACTION_REPS = [
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
    "ABSOLUTE",
]

CANONICAL_TASKS: dict[int, str] = {
    0: (
        "Use only your right hand to grasp the door handle. "
        "Keep your left arm still."
    ),
    1: (
        "Use only your right hand to rotate the door handle to unlock. "
        "Keep your left arm still."
    ),
    2: (
        "Use only your right hand to pull the cabinet door open. "
        "Keep your left arm still."
    ),
}

AUTO_EXECUTE_ARMS_BY_TASK_INDEX = {
    0: ("right",),
    1: ("right",),
    2: ("right",),
}

EXPECTED_NEW_EMBODIMENT = {
    "video_keys": VIDEO_KEYS,
    "state_keys": STATE_KEYS,
    "action_keys": ACTION_KEYS,
    "language_keys": [LANGUAGE_KEY],
    "action_horizon": ACTION_HORIZON,
    "action_reps": EXPECTED_ACTION_REPS,
    "observation_delta_indices": OBSERVATION_DELTA_INDICES,
    "state_history_length": STATE_HISTORY_LENGTH,
}
