"""Decode physical absolute actions from ``policy.get_action`` outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from quanta_x1_inference.constants import (
    ACTION_HORIZON,
    DEFAULT_EXECUTION_HORIZON,
    EXPECTED_NEW_EMBODIMENT,
    STATE_DIMS,
)

ACTION_KEYS = list(EXPECTED_NEW_EMBODIMENT["action_keys"])

# Quanta X1 SDK right-arm joint limits (rad), J1..J6.
SDK_RIGHT_ARM_LOWER = np.array(
    [-2.792, 0.0, -3.14, -1.57, -1.4, -1.745],
    dtype=np.float32,
)
SDK_RIGHT_ARM_UPPER = np.array(
    [2.792, 3.44, 0.0, 1.57, 1.4, 1.745],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DecodedAction:
    """One decoded absolute action step (physical units, train parquet layout)."""

    eef_9d: np.ndarray
    gripper_position: float
    joint_position: np.ndarray
    vector16: np.ndarray

    @property
    def sdk_joint_targets(self) -> np.ndarray:
        return clip_joints_to_sdk(self.joint_position)


def action_keys(modality_configs: dict[str, Any] | None = None) -> list[str]:
    if modality_configs is None:
        return list(ACTION_KEYS)
    return list(modality_configs["action"].modality_keys)


def unbatch_policy_action(
    action_dict: dict[str, np.ndarray],
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> dict[str, np.ndarray]:
    """Strip batch dim from ``policy.get_action`` output -> ``{key: (T, D)}``."""
    keys = list(action_keys_list or action_dict.keys())
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in action_dict:
            raise KeyError(f"Missing action key {key!r} in policy output")
        arr = np.asarray(action_dict[key], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[batch_index]
        elif arr.ndim != 2:
            raise ValueError(f"action[{key}] expected (B,T,D) or (T,D), got {arr.shape}")
        out[key] = arr
    return out


def components_to_vector16(
    *,
    eef_9d: np.ndarray,
    gripper_position: np.ndarray | float,
    joint_position: np.ndarray,
) -> np.ndarray:
    eef = np.asarray(eef_9d, dtype=np.float32).reshape(-1)
    grip = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    joints = np.asarray(joint_position, dtype=np.float32).reshape(-1)
    if eef.shape != (STATE_DIMS["eef_9d"],):
        raise ValueError(f"eef_9d shape {eef.shape}, expected ({STATE_DIMS['eef_9d']},)")
    if grip.shape != (STATE_DIMS["gripper_position"],):
        raise ValueError(
            f"gripper_position shape {grip.shape}, expected ({STATE_DIMS['gripper_position']},)"
        )
    if joints.shape != (STATE_DIMS["joint_position"],):
        raise ValueError(
            f"joint_position shape {joints.shape}, expected ({STATE_DIMS['joint_position']},)"
        )
    return np.concatenate([eef, grip, joints], axis=0).astype(np.float32)


def vector16_to_decoded(vector16: np.ndarray) -> DecodedAction:
    vec = np.asarray(vector16, dtype=np.float32).reshape(-1)
    if vec.shape != (16,):
        raise ValueError(f"expected 16D vector, got {vec.shape}")
    eef = vec[0:9]
    grip = float(vec[9])
    joints = vec[10:16]
    return DecodedAction(
        eef_9d=eef,
        gripper_position=grip,
        joint_position=joints,
        vector16=vec,
    )


def decode_action_at_step(
    action_dict: dict[str, np.ndarray],
    step_index: int,
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> DecodedAction:
    """Decode one horizon index from decoded ``policy.get_action`` output."""
    unbatched = unbatch_policy_action(
        action_dict,
        action_keys_list=action_keys_list,
        batch_index=batch_index,
    )
    horizon = next(iter(unbatched.values())).shape[0]
    if step_index < 0 or step_index >= horizon:
        raise IndexError(f"step_index {step_index} out of range for horizon {horizon}")

    parts = {key: unbatched[key][step_index] for key in unbatched}
    vector16 = components_to_vector16(
        eef_9d=parts["eef_9d"],
        gripper_position=parts["gripper_position"],
        joint_position=parts["joint_position"],
    )
    return vector16_to_decoded(vector16)


def decode_execution_horizon(
    action_dict: dict[str, np.ndarray],
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> list[DecodedAction]:
    """Decode the first ``execution_horizon`` steps (open-loop stride contract)."""
    if execution_horizon < 1:
        raise ValueError("execution_horizon must be >= 1")
    unbatched = unbatch_policy_action(
        action_dict,
        action_keys_list=action_keys_list,
        batch_index=batch_index,
    )
    horizon = next(iter(unbatched.values())).shape[0]
    if execution_horizon > horizon:
        raise ValueError(
            f"execution_horizon={execution_horizon} exceeds model horizon={horizon}"
        )
    return [
        decode_action_at_step(
            action_dict,
            step_index=j,
            action_keys_list=action_keys_list,
            batch_index=batch_index,
        )
        for j in range(execution_horizon)
    ]


def decode_open_loop_chunk(
    action_dict: dict[str, np.ndarray],
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> np.ndarray:
    """Return ``(execution_horizon, 16)`` vectors matching ``open_loop_eval`` concat."""
    decoded = decode_execution_horizon(
        action_dict,
        execution_horizon,
        action_keys_list=action_keys_list,
        batch_index=batch_index,
    )
    return np.stack([item.vector16 for item in decoded], axis=0)


def open_loop_concat_at_step(
    action_dict: dict[str, np.ndarray],
    step_index: int,
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> np.ndarray:
    """Same concat as ``open_loop_eval.parse_action_gr00t`` + per-step slice."""
    keys = list(action_keys_list or ACTION_KEYS)
    unbatched = unbatch_policy_action(
        action_dict,
        action_keys_list=keys,
        batch_index=batch_index,
    )
    return np.concatenate(
        [np.atleast_1d(unbatched[key][step_index]) for key in keys],
        axis=0,
    ).astype(np.float32)


def clip_joints_to_sdk(joints: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(joints, dtype=np.float32).reshape(6),
        SDK_RIGHT_ARM_LOWER,
        SDK_RIGHT_ARM_UPPER,
    ).astype(np.float32)


def validate_execution_horizon(
    modality_configs: dict[str, Any],
    execution_horizon: int,
) -> int:
    """Fail fast using the same contract as open-loop eval."""
    from gr00t.eval._horizon_contract import PolicyHorizonSpec

    spec = PolicyHorizonSpec.from_modality_config(
        modality_configs,
        n_action_steps=execution_horizon,
    )
    return spec.n_action_steps


def max_action_horizon(modality_configs: dict[str, Any] | None = None) -> int:
    if modality_configs is None:
        return ACTION_HORIZON
    return len(modality_configs["action"].delta_indices)
