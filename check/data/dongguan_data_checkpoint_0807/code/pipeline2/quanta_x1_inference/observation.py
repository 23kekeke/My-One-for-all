"""Build GR00T observations (train-aligned flat + nested batched forms)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quanta_x1_inference.constants import (
    EXPECTED_NEW_EMBODIMENT,
    STATE_DIMS,
    TASK_TEXT,
)

LANGUAGE_KEY = EXPECTED_NEW_EMBODIMENT["language_keys"][0]
VIDEO_KEYS = EXPECTED_NEW_EMBODIMENT["video_keys"]
STATE_KEYS = EXPECTED_NEW_EMBODIMENT["state_keys"]

FLAT_STATE_KEYS = [f"state.{k}" for k in STATE_KEYS]
FLAT_VIDEO_KEYS = [f"video.{k}" for k in VIDEO_KEYS]


def inference_modality_configs(modality_configs: dict[str, Any]) -> dict[str, Any]:
    """Modality configs for policy input (no action labels)."""
    configs = dict(modality_configs)
    configs.pop("action", None)
    return configs


def _as_state_row(values: np.ndarray | float | list[float], *, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"expected state dim {dim}, got shape {arr.shape}")
    return arr.reshape(1, dim)


def build_flat_observation(
    *,
    head_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    eef_9d: np.ndarray,
    gripper_position: np.ndarray | float | list[float],
    joint_position: np.ndarray | list[float],
    task_text: str = TASK_TEXT,
) -> dict[str, Any]:
    """Build flat ``{modality.key: value}`` observation (pre-``parse_observation_gr00t``).

    Shapes match ``open_loop_eval.evaluate_single_trajectory``:
      - state.* : (T=1, D)
      - video.* : (T=1, H, W, C) uint8 RGB
      - language: str
    """
    head = np.asarray(head_camera)
    wrist = np.asarray(right_arm_camera)
    if head.ndim != 3:
        raise ValueError(f"head_camera must be HWC, got shape {head.shape}")
    if wrist.ndim != 3:
        raise ValueError(f"right_arm_camera must be HWC, got shape {wrist.shape}")
    if head.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError("camera arrays must be uint8 RGB")

    return {
        "video.head_camera": np.asarray([head], dtype=np.uint8),
        "video.right_arm_camera": np.asarray([wrist], dtype=np.uint8),
        "state.eef_9d": _as_state_row(eef_9d, dim=STATE_DIMS["eef_9d"]),
        "state.gripper_position": _as_state_row(gripper_position, dim=STATE_DIMS["gripper_position"]),
        "state.joint_position": _as_state_row(joint_position, dim=STATE_DIMS["joint_position"]),
        LANGUAGE_KEY: str(task_text),
    }


def flat_observation_from_step_data(step_data: Any) -> dict[str, Any]:
    """Convert ``VLAStepData`` from ``extract_step_data`` to flat observation dict."""
    obs: dict[str, Any] = {}
    for key, value in step_data.states.items():
        obs[f"state.{key}"] = np.asarray(value, dtype=np.float32)
    for key, value in step_data.images.items():
        obs[f"video.{key}"] = np.array(value, dtype=np.uint8)
    obs[LANGUAGE_KEY] = step_data.text
    return obs


def build_observation(
    flat_obs: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Parse flat observation into nested batched policy input."""
    from gr00t.data.utils import parse_observation_gr00t

    configs = inference_modality_configs(modality_configs)
    return parse_observation_gr00t(flat_obs, configs)


def build_observation_from_components(
    *,
    head_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    eef_9d: np.ndarray,
    gripper_position: np.ndarray | float | list[float],
    joint_position: np.ndarray | list[float],
    task_text: str = TASK_TEXT,
    modality_configs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build flat + parsed observation for ``policy.get_action``."""
    flat = build_flat_observation(
        head_camera=head_camera,
        right_arm_camera=right_arm_camera,
        eef_9d=eef_9d,
        gripper_position=gripper_position,
        joint_position=joint_position,
        task_text=task_text,
    )
    parsed = build_observation(flat, modality_configs)
    return flat, parsed


def vector16_to_components(state_16d: np.ndarray) -> dict[str, np.ndarray]:
    """Split 16D vector into modality components (dataset / parquet layout)."""
    vec = np.asarray(state_16d, dtype=np.float32).reshape(-1)
    if vec.shape != (16,):
        raise ValueError(f"expected 16D state, got shape {vec.shape}")
    return {
        "eef_9d": vec[0:9],
        "gripper_position": vec[9:10],
        "joint_position": vec[10:16],
    }


def build_flat_observation_from_state16(
    *,
    head_camera: np.ndarray,
    right_arm_camera: np.ndarray,
    state_16d: np.ndarray,
    task_text: str = TASK_TEXT,
) -> dict[str, Any]:
    parts = vector16_to_components(state_16d)
    return build_flat_observation(
        head_camera=head_camera,
        right_arm_camera=right_arm_camera,
        eef_9d=parts["eef_9d"],
        gripper_position=parts["gripper_position"],
        joint_position=parts["joint_position"],
        task_text=task_text,
    )


def compare_flat_observations(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare two flat observations key-by-key."""
    keys = sorted(set(expected) | set(actual))
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for key in keys:
        if key not in expected:
            errors.append(f"missing in expected: {key}")
            checks.append({"key": key, "ok": False, "reason": "missing_expected"})
            continue
        if key not in actual:
            errors.append(f"missing in actual: {key}")
            checks.append({"key": key, "ok": False, "reason": "missing_actual"})
            continue

        exp = expected[key]
        act = actual[key]
        if isinstance(exp, str):
            ok = exp == act
            checks.append({"key": key, "ok": ok, "expected": exp, "actual": act})
            if not ok:
                errors.append(f"{key}: text mismatch")
            continue

        exp_arr = np.asarray(exp)
        act_arr = np.asarray(act)
        if exp_arr.shape != act_arr.shape:
            checks.append(
                {
                    "key": key,
                    "ok": False,
                    "expected_shape": list(exp_arr.shape),
                    "actual_shape": list(act_arr.shape),
                }
            )
            errors.append(f"{key}: shape {act_arr.shape} != {exp_arr.shape}")
            continue

        if exp_arr.dtype == np.uint8 or act_arr.dtype == np.uint8:
            ok = np.array_equal(exp_arr, act_arr)
            max_diff = 0.0 if ok else float(np.max(np.abs(exp_arr.astype(np.int16) - act_arr.astype(np.int16))))
        else:
            max_diff = float(np.max(np.abs(exp_arr.astype(np.float64) - act_arr.astype(np.float64))))
            ok = max_diff <= atol

        checks.append({"key": key, "ok": ok, "max_abs_diff": max_diff})
        if not ok:
            errors.append(f"{key}: max_abs_diff={max_diff:.3e} > {atol:.3e}")

    ok = not errors
    return {"ok": ok, "checks": checks, "errors": errors}


def compare_parsed_observations(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    def flatten_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for modality in ("video", "state"):
            for key, value in parsed.get(modality, {}).items():
                flat[f"{modality}.{key}"] = value
        for key, value in parsed.get("language", {}).items():
            flat[key] = value[0][0] if isinstance(value, list) and value and isinstance(value[0], list) else value
        return flat

    return compare_flat_observations(flatten_parsed(expected), flatten_parsed(actual), atol=atol)
