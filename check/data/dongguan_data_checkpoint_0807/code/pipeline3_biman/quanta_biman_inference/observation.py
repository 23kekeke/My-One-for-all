"""Build GR00T 32D biman observations (train-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quanta_biman_config import OBSERVATION_DELTA_INDICES, STATE_HISTORY_LENGTH
from quanta_biman_inference.constants import (
    CANONICAL_TASKS,
    LANGUAGE_KEY,
    STATE_DIM,
    STATE_DIMS,
    STATE_KEYS,
    VIDEO_KEYS,
)


def inference_modality_configs(modality_configs: dict[str, Any]) -> dict[str, Any]:
    configs = dict(modality_configs)
    configs.pop("action", None)
    return configs


def resolve_padded_step_indices(
    step_index: int,
    episode_length: int,
    delta_indices: list[int] | None = None,
) -> list[int]:
    """Match ``extract_step_data(..., allow_padding=True)`` index selection."""
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    deltas = list(delta_indices or OBSERVATION_DELTA_INDICES)
    last = episode_length - 1
    return [
        max(0, min(step_index + delta, last))
        for delta in deltas
    ]


def _validate_uint8_hwc(name: str, img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be HWC, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB")
    return arr


def _as_state_stack(
    values: list[np.ndarray | float | list[float]],
    *,
    dim: int,
) -> np.ndarray:
    rows = [np.asarray(v, dtype=np.float32).reshape(-1) for v in values]
    stacked = np.stack(rows, axis=0)
    if stacked.shape != (len(values), dim):
        raise ValueError(f"expected state stack {(len(values), dim)}, got {stacked.shape}")
    return stacked


@dataclass(frozen=True)
class BimanSnapshot:
    head_camera: np.ndarray
    left_arm_camera: np.ndarray
    right_arm_camera: np.ndarray
    left_eef_9d: np.ndarray
    left_gripper_position: float
    left_joint_position: np.ndarray
    right_eef_9d: np.ndarray
    right_gripper_position: float
    right_joint_position: np.ndarray


class SparseTemporalBuffer:
    """Live buffer: one snapshot per control cycle; sparse [-5, 0] at infer time."""

    def __init__(self, delta_indices: list[int] | None = None) -> None:
        self.delta_indices = list(delta_indices or OBSERVATION_DELTA_INDICES)
        self._snapshots: list[BimanSnapshot] = []

    def append(self, snapshot: BimanSnapshot) -> None:
        self._snapshots.append(snapshot)

    def clear(self) -> None:
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self._snapshots)

    def build_flat_observation(self, task_text: str) -> dict[str, Any]:
        if not self._snapshots:
            raise RuntimeError("SparseTemporalBuffer is empty")
        step_index = len(self._snapshots) - 1
        indices = resolve_padded_step_indices(
            step_index,
            len(self._snapshots),
            self.delta_indices,
        )
        return build_flat_observation_from_snapshots(
            [self._snapshots[i] for i in indices],
            task_text=task_text,
        )


def build_flat_observation_from_snapshots(
    snapshots: list[BimanSnapshot],
    *,
    task_text: str,
) -> dict[str, Any]:
    expected_t = len(OBSERVATION_DELTA_INDICES)
    if len(snapshots) != expected_t:
        raise ValueError(
            f"expected {expected_t} temporal snapshots for {OBSERVATION_DELTA_INDICES}, "
            f"got {len(snapshots)}"
        )

    head = np.stack(
        [_validate_uint8_hwc("head_camera", s.head_camera) for s in snapshots],
        axis=0,
    )
    left_cam = np.stack(
        [_validate_uint8_hwc("left_arm_camera", s.left_arm_camera) for s in snapshots],
        axis=0,
    )
    right_cam = np.stack(
        [_validate_uint8_hwc("right_arm_camera", s.right_arm_camera) for s in snapshots],
        axis=0,
    )

    return {
        "video.head_camera": head,
        "video.left_arm_camera": left_cam,
        "video.right_arm_camera": right_cam,
        "state.left_eef_9d": _as_state_stack(
            [s.left_eef_9d for s in snapshots], dim=STATE_DIMS["eef_9d"]
        ),
        "state.left_gripper_position": _as_state_stack(
            [s.left_gripper_position for s in snapshots],
            dim=STATE_DIMS["gripper_position"],
        ),
        "state.left_joint_position": _as_state_stack(
            [s.left_joint_position for s in snapshots],
            dim=STATE_DIMS["joint_position"],
        ),
        "state.right_eef_9d": _as_state_stack(
            [s.right_eef_9d for s in snapshots], dim=STATE_DIMS["eef_9d"]
        ),
        "state.right_gripper_position": _as_state_stack(
            [s.right_gripper_position for s in snapshots],
            dim=STATE_DIMS["gripper_position"],
        ),
        "state.right_joint_position": _as_state_stack(
            [s.right_joint_position for s in snapshots],
            dim=STATE_DIMS["joint_position"],
        ),
        LANGUAGE_KEY: str(task_text),
    }


def arm16_to_components(arm16: np.ndarray) -> dict[str, np.ndarray]:
    vec = np.asarray(arm16, dtype=np.float32).reshape(-1)
    if vec.shape != (16,):
        raise ValueError(f"expected 16D arm vector, got {vec.shape}")
    return {
        "eef_9d": vec[0:9],
        "gripper_position": vec[9:10],
        "joint_position": vec[10:16],
    }


def components_to_arm16(
    *,
    eef_9d: np.ndarray,
    gripper_position: np.ndarray | float,
    joint_position: np.ndarray,
) -> np.ndarray:
    eef = np.asarray(eef_9d, dtype=np.float32).reshape(-1)
    grip = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    joints = np.asarray(joint_position, dtype=np.float32).reshape(-1)
    if eef.shape != (STATE_DIMS["eef_9d"],):
        raise ValueError(f"eef_9d shape {eef.shape}")
    if grip.shape != (STATE_DIMS["gripper_position"],):
        raise ValueError(f"gripper_position shape {grip.shape}")
    if joints.shape != (STATE_DIMS["joint_position"],):
        raise ValueError(f"joint_position shape {joints.shape}")
    return np.concatenate([eef, grip, joints], axis=0).astype(np.float32)


def vector32_to_components(state_32d: np.ndarray) -> dict[str, np.ndarray]:
    vec = np.asarray(state_32d, dtype=np.float32).reshape(-1)
    if vec.shape != (STATE_DIM,):
        raise ValueError(f"expected 32D state, got shape {vec.shape}")
    left = arm16_to_components(vec[0:16])
    right = arm16_to_components(vec[16:32])
    return {
        "left_eef_9d": left["eef_9d"],
        "left_gripper_position": left["gripper_position"],
        "left_joint_position": left["joint_position"],
        "right_eef_9d": right["eef_9d"],
        "right_gripper_position": right["gripper_position"],
        "right_joint_position": right["joint_position"],
    }


def snapshot_from_components(
    *,
    head_camera: np.ndarray,
    left_arm_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    left_eef_9d: np.ndarray,
    left_gripper_position: np.ndarray | float | list[float],
    left_joint_position: np.ndarray | list[float],
    right_eef_9d: np.ndarray,
    right_gripper_position: np.ndarray | float | list[float],
    right_joint_position: np.ndarray | list[float],
) -> BimanSnapshot:
    return BimanSnapshot(
        head_camera=_validate_uint8_hwc("head_camera", head_camera),
        left_arm_camera=_validate_uint8_hwc("left_arm_camera", left_arm_camera),
        right_arm_camera=_validate_uint8_hwc("right_arm_camera", right_arm_camera),
        left_eef_9d=np.asarray(left_eef_9d, dtype=np.float32).reshape(-1),
        left_gripper_position=float(left_gripper_position),
        left_joint_position=np.asarray(left_joint_position, dtype=np.float32).reshape(-1),
        right_eef_9d=np.asarray(right_eef_9d, dtype=np.float32).reshape(-1),
        right_gripper_position=float(right_gripper_position),
        right_joint_position=np.asarray(right_joint_position, dtype=np.float32).reshape(-1),
    )


def build_flat_observation(
    *,
    head_camera: np.ndarray,
    left_arm_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    left_eef_9d: np.ndarray,
    left_gripper_position: np.ndarray | float | list[float],
    left_joint_position: np.ndarray | list[float],
    right_eef_9d: np.ndarray,
    right_gripper_position: np.ndarray | float | list[float],
    right_joint_position: np.ndarray | list[float],
    task_text: str,
) -> dict[str, Any]:
    """Single-timestep convenience: pads both sparse slots with the same snapshot."""
    buffer = SparseTemporalBuffer()
    buffer.append(
        snapshot_from_components(
            head_camera=head_camera,
            left_arm_camera=left_arm_camera,
            right_arm_camera=right_arm_camera,
            left_eef_9d=left_eef_9d,
            left_gripper_position=left_gripper_position,
            left_joint_position=left_joint_position,
            right_eef_9d=right_eef_9d,
            right_gripper_position=right_gripper_position,
            right_joint_position=right_joint_position,
        )
    )
    return buffer.build_flat_observation(task_text)


def build_observation(
    flat_obs: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    from gr00t.data.utils import parse_observation_gr00t

    return parse_observation_gr00t(flat_obs, inference_modality_configs(modality_configs))


def build_observation_from_buffer(
    buffer: SparseTemporalBuffer,
    *,
    task_text: str,
    modality_configs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    flat = buffer.build_flat_observation(task_text)
    parsed = build_observation(flat, modality_configs)
    return flat, parsed


def build_observation_from_components(
    *,
    head_camera: np.ndarray,
    left_arm_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    left_eef_9d: np.ndarray,
    left_gripper_position: np.ndarray | float | list[float],
    left_joint_position: np.ndarray | list[float],
    right_eef_9d: np.ndarray,
    right_gripper_position: np.ndarray | float | list[float],
    right_joint_position: np.ndarray | list[float],
    task_text: str,
    modality_configs: dict[str, Any],
    temporal_buffer: SparseTemporalBuffer | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = snapshot_from_components(
        head_camera=head_camera,
        left_arm_camera=left_arm_camera,
        right_arm_camera=right_arm_camera,
        left_eef_9d=left_eef_9d,
        left_gripper_position=left_gripper_position,
        left_joint_position=left_joint_position,
        right_eef_9d=right_eef_9d,
        right_gripper_position=right_gripper_position,
        right_joint_position=right_joint_position,
    )
    if temporal_buffer is None:
        buffer = SparseTemporalBuffer()
        buffer.append(snapshot)
    else:
        buffer = temporal_buffer
        buffer.append(snapshot)
    return build_observation_from_buffer(
        buffer,
        task_text=task_text,
        modality_configs=modality_configs,
    )


def build_flat_observation_from_state32(
    *,
    head_camera: np.ndarray,
    left_arm_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    state_32d: np.ndarray,
    task_text: str,
) -> dict[str, Any]:
    parts = vector32_to_components(state_32d)
    return build_flat_observation(
        head_camera=head_camera,
        left_arm_camera=left_arm_camera,
        right_arm_camera=right_arm_camera,
        left_eef_9d=parts["left_eef_9d"],
        left_gripper_position=parts["left_gripper_position"],
        left_joint_position=parts["left_joint_position"],
        right_eef_9d=parts["right_eef_9d"],
        right_gripper_position=parts["right_gripper_position"],
        right_joint_position=parts["right_joint_position"],
        task_text=task_text,
    )


def task_text_for_index(task_index: int) -> str:
    if task_index not in CANONICAL_TASKS:
        raise KeyError(f"Unknown task_index {task_index}; expected one of {sorted(CANONICAL_TASKS)}")
    return CANONICAL_TASKS[task_index]


def validate_state_keys(modality_configs: dict[str, Any]) -> None:
    state_keys = list(modality_configs["state"].modality_keys)
    if state_keys != STATE_KEYS:
        raise RuntimeError(f"Unexpected state keys {state_keys!r}, expected {STATE_KEYS!r}")
    video_keys = list(modality_configs["video"].modality_keys)
    if video_keys != VIDEO_KEYS:
        raise RuntimeError(f"Unexpected video keys {video_keys!r}, expected {VIDEO_KEYS!r}")


def validate_temporal_config(modality_configs: dict[str, Any]) -> None:
    video_deltas = list(modality_configs["video"].delta_indices)
    state_deltas = list(modality_configs["state"].delta_indices)
    if video_deltas != OBSERVATION_DELTA_INDICES:
        raise RuntimeError(
            f"video.delta_indices {video_deltas!r} != expected {OBSERVATION_DELTA_INDICES!r}"
        )
    if state_deltas != OBSERVATION_DELTA_INDICES:
        raise RuntimeError(
            f"state.delta_indices {state_deltas!r} != expected {OBSERVATION_DELTA_INDICES!r}"
        )
    if len(video_deltas) != STATE_HISTORY_LENGTH:
        raise RuntimeError(
            f"temporal length {len(video_deltas)} != STATE_HISTORY_LENGTH {STATE_HISTORY_LENGTH}"
        )
