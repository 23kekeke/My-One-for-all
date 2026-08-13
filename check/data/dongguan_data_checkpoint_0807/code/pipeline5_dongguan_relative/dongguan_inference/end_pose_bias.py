"""Task0-only end_pose pitch-up bias applied right before SDK commands.

Compensates aged gripper tip orientation. Does not change policy outputs on disk;
only rewrites orientation_xyzw sent to set_end_pose / execute_end_pose_trajectory.

Omit --task0-end-pose-pitch-up-deg (or pass 0) → no compensation (original behavior).
Task1/2 never apply.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

# Positive = tip gripper "up" (sign calibrated for Dongguan right-arm SDK).
# Default 0: no compensation unless CLI explicitly sets a non-zero angle.
DEFAULT_TASK0_END_POSE_PITCH_UP_DEG = 0.0

_ACTIVE_TASK_INDEX: int | None = None
_PITCH_UP_DEG: float = 0.0
_PATCHED = False


def set_task0_pitch_context(
    *,
    task_index: int | None,
    pitch_up_deg: float | None = None,
) -> None:
    global _ACTIVE_TASK_INDEX, _PITCH_UP_DEG
    _ACTIVE_TASK_INDEX = None if task_index is None else int(task_index)
    # None / omitted → no compensation.
    _PITCH_UP_DEG = 0.0 if pitch_up_deg is None else float(pitch_up_deg)


def should_apply(*, task_index: int | None = None, arm: str = "right") -> bool:
    ti = _ACTIVE_TASK_INDEX if task_index is None else int(task_index)
    return ti == 0 and arm == "right" and abs(_PITCH_UP_DEG) > 1e-9


def pitch_up_orientation_xyzw(
    orientation_xyzw: dict[str, Any],
    *,
    pitch_up_deg: float | None = None,
) -> dict[str, float]:
    """Tip gripper "up" by ``pitch_up_deg`` (positive = up).

    Empirically on this Dongguan right-arm SDK frame, body-fixed +Y pitch tips
    *down*; so we apply ``R_new = R_old @ R_y(-pitch_up_deg)``.
    """
    deg = float(_PITCH_UP_DEG if pitch_up_deg is None else pitch_up_deg)
    quat = np.array(
        [
            float(orientation_xyzw["x"]),
            float(orientation_xyzw["y"]),
            float(orientation_xyzw["z"]),
            float(orientation_xyzw["w"]),
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(quat))
    if n <= 0:
        raise ValueError("zero quaternion")
    quat = quat / n
    r_old = Rotation.from_quat(quat)
    # Negate: +deg CLI means tip up on this robot.
    r_delta = Rotation.from_euler("y", np.deg2rad(-deg))
    q_new = (r_old * r_delta).as_quat()  # xyzw
    return {
        "x": float(q_new[0]),
        "y": float(q_new[1]),
        "z": float(q_new[2]),
        "w": float(q_new[3]),
    }


def pitch_up_end_pose_dict(
    end_pose: dict[str, Any],
    *,
    pitch_up_deg: float | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(end_pose)
    if "orientation_xyzw" in out:
        out["orientation_xyzw"] = pitch_up_orientation_xyzw(
            out["orientation_xyzw"], pitch_up_deg=pitch_up_deg
        )
    elif "orientation" in out:
        out["orientation"] = pitch_up_orientation_xyzw(
            out["orientation"], pitch_up_deg=pitch_up_deg
        )
    else:
        raise KeyError("end_pose missing orientation_xyzw/orientation")
    return out


def maybe_pitch_up_end_pose(
    end_pose: dict[str, Any],
    *,
    task_index: int | None = None,
    arm: str = "right",
    pitch_up_deg: float | None = None,
) -> dict[str, Any]:
    if not should_apply(task_index=task_index, arm=arm):
        return end_pose
    return pitch_up_end_pose_dict(end_pose, pitch_up_deg=pitch_up_deg)


def install_live_end_pose_waypoint_patch() -> None:
    """Rewrite right-arm waypoints for task0 just before SDK trajectory execute."""
    global _PATCHED
    if _PATCHED:
        return

    import quanta_biman_inference.live_runner as lr

    orig = lr.policy_end_pose_waypoints_for_arm

    def patched(planned_steps: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
        waypoints = orig(planned_steps, arm)
        if not should_apply(arm=arm):
            return waypoints
        biased: list[dict[str, Any]] = []
        for wp in waypoints:
            item = dict(wp)
            item["end_pose"] = pitch_up_end_pose_dict(wp["end_pose"])
            biased.append(item)
        return biased

    lr.policy_end_pose_waypoints_for_arm = patched
    _PATCHED = True
    print(
        "[pipeline5] task0 right end_pose pitch-up hook installed "
        "(inactive until --task0-end-pose-pitch-up-deg DEG is set)",
        flush=True,
    )
