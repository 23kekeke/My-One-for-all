"""Train/infer contract for Dongguan multi345 relative GR00T."""

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
            if (ancestor / "code" / "pipeline5_dongguan_relative").is_dir():
                return ancestor.resolve()
    return None


_DEPLOY_ROOT = _detect_deploy_root()

if _DEPLOY_ROOT is not None:
    DEPLOY_ROOT = _DEPLOY_ROOT
    CODE_ROOT = DEPLOY_ROOT / "code"
    GR00T_REPO = CODE_ROOT / "isaacGr00t"
    PIPELINE2 = CODE_ROOT / "pipeline2"
    PIPELINE3 = CODE_ROOT / "pipeline3_biman"
    PIPELINE5 = CODE_ROOT / "pipeline5_dongguan_relative"
    ROOT = CODE_ROOT
    BASE_MODEL = DEPLOY_ROOT / "models/GR00T-N1.7-3B"
    COSMOS_MODEL = DEPLOY_ROOT / "models/Cosmos-Reason2-2B"
    CHECKPOINT_ROOT = DEPLOY_ROOT
    DEFAULT_CHECKPOINT = DEPLOY_ROOT / "checkpoints/multi345_v1/checkpoint-15000"
    DEFAULT_TRAIN_OUTPUT = DEPLOY_ROOT / "checkpoints"
    MULTI_DATASET = DEPLOY_ROOT / "lerobot/multi_345"
    INFERENCE_TMP = DEPLOY_ROOT / "tmp/dongguan_inference"
    DEFAULT_PER_TASK_HOME = DEPLOY_ROOT / "per_task_home.json"
else:
    DEPLOY_ROOT = None
    ROOT = Path(__file__).resolve().parents[2]
    GR00T_REPO = ROOT / "isaacGr00t"
    PIPELINE2 = ROOT / "pipeline2"
    PIPELINE3 = ROOT / "pipeline3_biman"
    PIPELINE5 = Path(__file__).resolve().parents[1]
    BASE_MODEL = ROOT / "GR00T-N1.7-3B"
    COSMOS_MODEL = ROOT / "Cosmos-Reason2-2B"
    CHECKPOINT_ROOT = Path("/data/dongguan_data_checkpoint_0807")
    DEFAULT_TRAIN_OUTPUT = (
        CHECKPOINT_ROOT / "output/dongguan_multi345_relative_qlora_r32_vit_r32_bs2_ga4"
    )
    DEFAULT_CHECKPOINT = DEFAULT_TRAIN_OUTPUT / "checkpoint-15000"
    MULTI_DATASET = CHECKPOINT_ROOT / "lerobot/multi_345"
    INFERENCE_TMP = PIPELINE5 / "tmp/dongguan_inference"
    DEFAULT_PER_TASK_HOME = CHECKPOINT_ROOT / "per_task_home.json"
    candidate = PIPELINE5.parents[1] / "per_task_home.json"
    if candidate.is_file():
        DEFAULT_PER_TASK_HOME = candidate

LIVE_RUNS_TMP = INFERENCE_TMP / "live_runs"
HOME_PREPOSITION_DEFAULT_MAX_JOINT_DELTA_RAD = 0.10  # unused for end_pose path (compat)
HOME_PREPOSITION_DEFAULT_TOLERANCE_RAD = 0.05  # unused for end_pose path (compat)
HOME_PREPOSITION_DEFAULT_TOLERANCE_M = 0.025  # SDK end_pose often lands ~2cm; allow 2.5cm
HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD = 0.20
HOME_PREPOSITION_DEFAULT_MAX_STEPS = 1  # forced: one slow extend to home; never retract
HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC = 1.0
HOME_PREPOSITION_DEFAULT_ARM_SETTLE_SEC = 1.0
HOME_PREPOSITION_DEFAULT_INTERPOLATE_HZ = 10.0
HOME_PREPOSITION_DEFAULT_MAX_LINEAR_SPEED_M_S = 0.015  # 1.5 cm/s
HOME_PREPOSITION_DEFAULT_MIN_DURATION_SEC = 5.0
HOME_PREPOSITION_DEFAULT_MAX_STEP_M = 0.008  # never command >8mm jump between waypoints
OPEN_LOOP_TMP = INFERENCE_TMP / "open_loop"
GR00T_PYTHON = GR00T_REPO / ".venv/bin/python"

EMBODIMENT_TAG_VALUE = "new_embodiment"
ACTION_HORIZON = 40
NUM_INFERENCE_TIMESTEPS = 4
DEFAULT_EXECUTION_HORIZON = 8
LIVE_DEFAULT_EXECUTION_HORIZON = 1
STATE_DIM = 32
TRAIN_FPS = 15.0

# Must match dongguan_relative_config.py
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
ACTION_KEYS = list(STATE_KEYS)

EXPECTED_ACTION_REPS = [
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
]

# task_index 0/1/2 = LeRobot task3/4/5 (grasp / rotate / pull)
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
