"""Decode 32D biman absolute actions from ``policy.get_action`` outputs.

Training action_configs may mark eef/joints as RELATIVE; after the policy
processor ``decode_action`` / unapply, values here are absolute in robot frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from quat_rot6d_utils import eef_9d_to_end_pose_dict

from quanta_biman_inference.constants import (
    ACTION_HORIZON,
    ACTION_KEYS,
    DEFAULT_EXECUTION_HORIZON,
    STATE_DIM,
    STATE_DIMS,
)
from quanta_biman_inference.observation import components_to_arm16

from quanta_biman_inference.sdk_joint_limits import (
    SDK_LEFT_ARM_LOWER,
    SDK_LEFT_ARM_UPPER,
    SDK_RIGHT_ARM_LOWER,
    SDK_RIGHT_ARM_UPPER,
    ArmName,
    clip_joints_to_sdk,
)


@dataclass(frozen=True)
class DecodedArmAction:
    arm: ArmName
    eef_9d: np.ndarray
    gripper_position: float
    joint_position: np.ndarray
    vector16: np.ndarray

    @property
    def sdk_joint_targets(self) -> np.ndarray:
        return clip_joints_to_sdk(self.arm, self.joint_position)


@dataclass(frozen=True)
class DecodedBimanAction:
    left: DecodedArmAction
    right: DecodedArmAction
    vector32: np.ndarray


def decoded_step_to_dict(decoded: DecodedBimanAction) -> dict[str, Any]:
    left_end_pose = eef_9d_to_end_pose_dict(decoded.left.eef_9d)
    right_end_pose = eef_9d_to_end_pose_dict(decoded.right.eef_9d)
    return {
        "vector32": decoded.vector32.astype(float).tolist(),
        "left": {
            "eef_9d": decoded.left.eef_9d.astype(float).tolist(),
            "end_pose": left_end_pose,
            "gripper_position": float(decoded.left.gripper_position),
            "joint_position_rad": decoded.left.joint_position.astype(float).tolist(),
            "sdk_joint_targets_rad": decoded.left.sdk_joint_targets.astype(float).tolist(),
        },
        "right": {
            "eef_9d": decoded.right.eef_9d.astype(float).tolist(),
            "end_pose": right_end_pose,
            "gripper_position": float(decoded.right.gripper_position),
            "joint_position_rad": decoded.right.joint_position.astype(float).tolist(),
            "sdk_joint_targets_rad": decoded.right.sdk_joint_targets.astype(float).tolist(),
        },
    }


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


def vector32_to_decoded(vector32: np.ndarray) -> DecodedBimanAction:
    vec = np.asarray(vector32, dtype=np.float32).reshape(-1)
    if vec.shape != (STATE_DIM,):
        raise ValueError(f"expected 32D vector, got {vec.shape}")

    left16 = vec[0:16]
    right16 = vec[16:32]
    left = _arm16_to_decoded("left", left16)
    right = _arm16_to_decoded("right", right16)
    return DecodedBimanAction(left=left, right=right, vector32=vec)


def _arm16_to_decoded(arm: ArmName, arm16: np.ndarray) -> DecodedArmAction:
    vec = np.asarray(arm16, dtype=np.float32).reshape(-1)
    if vec.shape != (16,):
        raise ValueError(f"expected 16D arm vector, got {vec.shape}")
    return DecodedArmAction(
        arm=arm,
        eef_9d=vec[0:9],
        gripper_position=float(vec[9]),
        joint_position=vec[10:16],
        vector16=vec,
    )


def decode_action_at_step(
    action_dict: dict[str, np.ndarray],
    step_index: int,
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> DecodedBimanAction:
    unbatched = unbatch_policy_action(
        action_dict,
        action_keys_list=action_keys_list,
        batch_index=batch_index,
    )
    horizon = next(iter(unbatched.values())).shape[0]
    if step_index < 0 or step_index >= horizon:
        raise IndexError(f"step_index {step_index} out of range for horizon {horizon}")

    left16 = components_to_arm16(
        eef_9d=unbatched["left_eef_9d"][step_index],
        gripper_position=unbatched["left_gripper_position"][step_index],
        joint_position=unbatched["left_joint_position"][step_index],
    )
    right16 = components_to_arm16(
        eef_9d=unbatched["right_eef_9d"][step_index],
        gripper_position=unbatched["right_gripper_position"][step_index],
        joint_position=unbatched["right_joint_position"][step_index],
    )
    return DecodedBimanAction(
        left=_arm16_to_decoded("left", left16),
        right=_arm16_to_decoded("right", right16),
        vector32=np.concatenate([left16, right16], axis=0),
    )


def decode_execution_horizon(
    action_dict: dict[str, np.ndarray],
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    *,
    action_keys_list: Sequence[str] | None = None,
    batch_index: int = 0,
) -> list[DecodedBimanAction]:
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


def validate_execution_horizon(
    modality_configs: dict[str, Any],
    execution_horizon: int,
) -> int:
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


def resolve_execute_arms(
    execute_arms: str,
    *,
    task_index: int | None,
) -> tuple[ArmName, ...]:
    mode = execute_arms.lower()
    if mode == "auto":
        if task_index is None:
            raise ValueError("--execute-arms auto requires --task-index")
        from quanta_biman_inference.constants import AUTO_EXECUTE_ARMS_BY_TASK_INDEX

        return tuple(AUTO_EXECUTE_ARMS_BY_TASK_INDEX[task_index])  # type: ignore[return-value]
    if mode == "right":
        return ("right",)
    if mode == "left":
        return ("left",)
    if mode == "both":
        return ("left", "right")
    if mode == "none":
        return ()
    raise ValueError(f"Unknown execute_arms={execute_arms!r}")
