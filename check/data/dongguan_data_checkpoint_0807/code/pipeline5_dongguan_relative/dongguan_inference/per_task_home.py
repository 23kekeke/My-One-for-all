"""Load Dongguan per-task home poses from deploy JSON (end_pose schema v2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dongguan_inference.constants import DEFAULT_PER_TASK_HOME


def resolve_per_task_home_path(path: Path | str | None = None) -> Path:
    if path is None:
        resolved = DEFAULT_PER_TASK_HOME
    else:
        resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"per_task_home JSON not found: {resolved}. "
            "Expected deploy root per_task_home.json (see README_per_task_home.md)."
        )
    return resolved.resolve()


def load_per_task_home_file(path: Path | str | None = None) -> dict[str, Any]:
    resolved = resolve_per_task_home_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "per_task_home" not in data or not isinstance(data["per_task_home"], list):
        raise ValueError(f"{resolved} missing per_task_home list")
    return data


def get_task_home(
    task_index: int,
    *,
    path: Path | str | None = None,
) -> dict[str, Any]:
    data = load_per_task_home_file(path)
    for entry in data["per_task_home"]:
        if int(entry["task_index"]) == int(task_index):
            return entry
    raise KeyError(f"task_index={task_index} not found in {resolve_per_task_home_path(path)}")


def arm_end_pose_targets(home: dict[str, Any], arm: str) -> tuple[dict[str, Any], float]:
    """Return (end_pose dict for SDK set_end_pose, gripper_position)."""
    arm_key = f"{arm}_arm"
    if arm_key not in home:
        raise KeyError(f"home missing {arm_key}")
    block = home[arm_key]
    if "end_pose" not in block:
        raise KeyError(
            f"{arm_key} missing end_pose (schema v2). "
            "per_task_home.json must use position_m + orientation_xyzw, not joint_position_rad."
        )
    end_pose = block["end_pose"]
    if "position_m" not in end_pose and "position" not in end_pose:
        raise KeyError(f"{arm_key}.end_pose missing position_m")
    if "orientation_xyzw" not in end_pose and "orientation" not in end_pose:
        raise KeyError(f"{arm_key}.end_pose missing orientation_xyzw")
    gripper = float(block["gripper_position"])
    return end_pose, gripper


def arm_joint_targets(home: dict[str, Any], arm: str) -> tuple[np.ndarray, float]:
    """Legacy joint home (schema v1). Prefer arm_end_pose_targets for schema v2."""
    arm_key = f"{arm}_arm"
    if arm_key not in home:
        raise KeyError(f"home missing {arm_key}")
    block = home[arm_key]
    if "joint_position_rad" not in block:
        raise KeyError(
            f"{arm_key} missing joint_position_rad; current per_task_home is end_pose-only. "
            "Use arm_end_pose_targets()."
        )
    joints = np.asarray(block["joint_position_rad"], dtype=np.float32).reshape(6)
    gripper = float(block["gripper_position"])
    return joints, gripper
