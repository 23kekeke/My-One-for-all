"""Train/inference contract constants for Quanta X1 16D (assert against checkpoint processor)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GR00T_REPO = ROOT / "isaacGr00t"
PIPELINE2 = ROOT / "pipeline2"
BASE_MODEL = ROOT / "GR00T-N1.7-3B"
COSMOS_MODEL = ROOT / "Cosmos-Reason2-2B"
DEFAULT_CHECKPOINT = (
    PIPELINE2 / "outputs/quanta_x1_qlora_bs4_ga2/checkpoint-5000"
)
VAL_DATASET = ROOT / "dataset/quanta_lerobot_val"
TRAIN_DATASET = ROOT / "dataset/quanta_lerobot_train"
FULL_DATASET = ROOT / "dataset/quanta_lerobot"
DEFAULT_TRAIN_OUTPUT = PIPELINE2 / "outputs/quanta_x1_qlora_bs4_ga2"
DEFAULT_SWEEP_CHECKPOINT_STEPS = (5000, 8000, 10000, 15000)
INFERENCE_TMP = PIPELINE2 / "tmp/quanta_x1_inference"
LIVE_RUNS_TMP = INFERENCE_TMP / "live_runs"
DEFAULT_LIVE_CHECKPOINT = DEFAULT_TRAIN_OUTPUT / "checkpoint-15000"

EMBODIMENT_TAG_VALUE = "new_embodiment"
ACTION_HORIZON = 40
NUM_INFERENCE_TIMESTEPS = 4
DEFAULT_EXECUTION_HORIZON = 8
TASK_TEXT = "Right hand presses the switch."

# 16D layout: eef_9d(9) + gripper(1) + joint(6)
STATE_DIMS = {
    "eef_9d": 9,
    "gripper_position": 1,
    "joint_position": 6,
}

EXPECTED_NEW_EMBODIMENT = {
    "video_keys": ["head_camera", "right_arm_camera"],
    "state_keys": ["eef_9d", "gripper_position", "joint_position"],
    "action_keys": ["eef_9d", "gripper_position", "joint_position"],
    "language_keys": ["annotation.language.language_instruction"],
    "action_horizon": ACTION_HORIZON,
    "action_reps": ["ABSOLUTE", "ABSOLUTE", "ABSOLUTE"],
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
