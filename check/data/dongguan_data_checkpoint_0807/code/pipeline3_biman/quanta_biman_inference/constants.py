"""Train/inference contract for Quanta X1 biman 32D GR00T."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GR00T_REPO = ROOT / "isaacGr00t"
PIPELINE2 = ROOT / "pipeline2"
PIPELINE3 = ROOT / "pipeline3_biman"
BASE_MODEL = ROOT / "GR00T-N1.7-3B"
COSMOS_MODEL = ROOT / "Cosmos-Reason2-2B"

DEFAULT_TRAIN_OUTPUT = PIPELINE3 / "outputs/quanta_biman_task1_qlora_bs4_ga2"
DEFAULT_CHECKPOINT = DEFAULT_TRAIN_OUTPUT / "checkpoint-15000"
TASK1_DATASET = ROOT / "dataset/quanta_biman_lerobot/task1"
TASK2_DATASET = ROOT / "dataset/quanta_biman_lerobot/task2"
VAL_DATASET = ROOT / "dataset/quanta_biman_lerobot_val"
TRAIN_DATASET = ROOT / "dataset/quanta_biman_lerobot_train"
FULL_DATASET = ROOT / "dataset/quanta_biman_lerobot"

INFERENCE_TMP = PIPELINE3 / "tmp/quanta_biman_inference"
LIVE_RUNS_TMP = INFERENCE_TMP / "live_runs"

EMBODIMENT_TAG_VALUE = "new_embodiment"
ACTION_HORIZON = 40
NUM_INFERENCE_TIMESTEPS = 4
# Open-loop eval default (dataset steps at 15fps). Live runner defaults to 1 (conservative).
DEFAULT_EXECUTION_HORIZON = 8
LIVE_DEFAULT_EXECUTION_HORIZON = 1
STATE_DIM = 32
ARM_DIM = 16

# Temporal observation (must match quanta_biman_config.py + checkpoint state_history_length).
OBSERVATION_DELTA_INDICES = [-5, 0]
STATE_HISTORY_LENGTH = len(OBSERVATION_DELTA_INDICES)
TRAIN_FPS = 15.0
# Training: delta -5 at 15fps → ~333ms between the two observation slots.
TEMPORAL_SPAN_SEC = abs(OBSERVATION_DELTA_INDICES[0]) / TRAIN_FPS
# Live: first N cycles pad earliest snapshot (same as train allow_padding=True).
TEMPORAL_WARMUP_CYCLES = abs(OBSERVATION_DELTA_INDICES[0])

# SDK capture/execute: state streams can lag after arm motion (transient UNAVAILABLE).
END_POSE_PRE_DELAY_SEC = 0.10
END_POSE_MAX_ATTEMPTS = 6
CAPTURE_SUBPROCESS_MAX_ATTEMPTS = 3
EXECUTE_SUBPROCESS_MAX_ATTEMPTS = 3

# SDK docs: control RPCs must stay <= 200 Hz. Interpolation runs on the dev machine (daemon).
MAX_SDK_CONTROL_HZ = 200.0
DEFAULT_EXECUTE_INTERPOLATE_HZ = 0.0

# Policy live execute via set_end_pose (faster than home preposition speeds).
DEFAULT_EXECUTE_VIA = "joint"  # Dongguan overrides to end_pose
POLICY_END_POSE_INTERPOLATE_HZ = 50.0
POLICY_END_POSE_MAX_LINEAR_SPEED_M_S = 0.08  # 8 cm/s
POLICY_END_POSE_MIN_DURATION_SEC = 0.0  # per-segment floor; train_fps also floors timing
POLICY_END_POSE_MAX_STEP_M = 0.02
POLICY_END_POSE_SETTLE_SEC = 0.0  # end-of-trajectory settle only (not per policy step)
DEFAULT_TRAJECTORY_SETTLE_SEC = 0.05

# Recommended live presets (task1-only, checkpoint-15000). Tune on robot after shadow.
LIVE_PRESETS: dict[str, dict[str, float | int]] = {
    "shadow_smoke": {
        "cycles": 1,
        "execution_horizon": 1,
        "interval_sec": 0.0,
        "settle_sec": 0.0,
        "max_joint_delta_rad": 0.05,
    },
    "shadow_warmup": {
        "cycles": 10,
        "execution_horizon": 1,
        "interval_sec": 0.0,
        "settle_sec": 0.0,
        "max_joint_delta_rad": 0.05,
    },
    "live_short": {
        "cycles": 5,
        "execution_horizon": 1,
        "interval_sec": 0.0,
        "settle_sec": 0.6,
        "max_joint_delta_rad": 0.05,
    },
    "live_task1": {
        "cycles": 80,
        "execution_horizon": 4,
        "interval_sec": 0.0,
        "settle_sec": 0.6,
        "max_joint_delta_rad": 0.08,
    },
}

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

# Per action group (left/right × eef, gripper, joint); matches quanta_biman_config.py.
EXPECTED_NEW_EMBODIMENT = {
    "video_keys": VIDEO_KEYS,
    "state_keys": STATE_KEYS,
    "action_keys": ACTION_KEYS,
    "language_keys": [LANGUAGE_KEY],
    "action_horizon": ACTION_HORIZON,
    "action_reps": ["ABSOLUTE"] * 6,
    "observation_delta_indices": OBSERVATION_DELTA_INDICES,
    "state_history_length": STATE_HISTORY_LENGTH,
}

CANONICAL_TASKS: dict[int, str] = {
    0: (
        "Extend only your right arm forward and press the button below the handle. "
        "Keep your left arm still."
    ),
    1: (
        "Use only your right hand to rotate the handle and open the door. "
        "Keep your left arm still."
    ),
    2: "Raise the left hand, and push the door to 90 degrees with the right hand.",
}

TASK_GROUP_BY_INDEX = {0: "task1", 1: "task2", 2: "task3"}

# Which arms to send to SDK when --execute-arms auto (matches training task semantics).
AUTO_EXECUTE_ARMS_BY_TASK_INDEX = {
    0: ("right",),
    1: ("right",),
    2: ("left", "right"),
}

# task2 LeRobot export (379 ep): mean right-arm pose at episode start.
# Method: steady window — avg obs over frames [0, min(motion_start, 15)) before first
# |action-state|>0.02 rad on right joints. See task2/meta/task2_start_pose_stats.json.
# right eef xyz mean ~ [0.074, 0.009, 0.197]; demos later reach x peak ~0.46.
TASK2_RIGHT_ARM_START_JOINTS: tuple[float, ...] = (
    0.0519,
    0.7817,
    -0.8234,
    -0.0145,
    -0.0158,
    0.0249,
)
TASK2_RIGHT_ARM_START_GRIPPER = -0.0240

# task2 hybrid live (plan A): first SDK move to handle zone (eef x >= 0.38 in training).
# Mean right-arm pose at first frame with x>=0.38 per episode (379 ep).
# right eef xyz mean ~ [0.382, -0.028, 0.206]
TASK2_NEAR_HANDLE_JOINTS: tuple[float, ...] = (
    -0.0474,
    2.0425,
    -2.0538,
    0.2325,
    -0.0563,
    0.0121,
)
TASK2_NEAR_HANDLE_GRIPPER = -0.0101
TASK2_NEAR_HANDLE_MIN_EEF_X_M = 0.36

TASK2_PREPOSITION_DEFAULT_MAX_JOINT_DELTA_RAD = 0.10
TASK2_PREPOSITION_DEFAULT_TOLERANCE_RAD = 0.05
TASK2_PREPOSITION_DEFAULT_MAX_STEPS = 25
TASK2_NEAR_HANDLE_DEFAULT_MAX_JOINT_DELTA_RAD = 0.12
TASK2_NEAR_HANDLE_DEFAULT_TOLERANCE_RAD = 0.08
TASK2_NEAR_HANDLE_DEFAULT_MAX_STEPS = 40

STATE_DIMS = {
    "eef_9d": 9,
    "gripper_position": 1,
    "joint_position": 6,
}

EXPECTED_PROCESSOR_KWARGS = {
    "letter_box_transform": True,
    "shortest_image_edge": 256,
    "crop_fraction": 1.0,
    "use_albumentations": True,
    "formalize_language": True,
    "use_percentiles": True,
    "max_action_horizon": ACTION_HORIZON,
}

LIVE_ACK_TOKEN = "QUANTA_BIMAN_32D_LIVE"
