"""Quaternion (xyzw) -> rot6d utilities for pipeline3_biman."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def orientation_dict_to_quat_xyzw(orientation: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to rot6d = first two rows of R, row-major."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 0:
        raise ValueError("zero quaternion")
    quat_xyzw = quat_xyzw / norm
    rotation_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    return rotation_matrix[:2, :].reshape(6)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert rot6d (first two rows flattened) to a 3x3 rotation matrix."""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row1 = rot6d[0]
    row2 = rot6d[1]
    n1 = float(np.linalg.norm(row1))
    if n1 <= 1e-12:
        raise ValueError("degenerate rot6d: zero first row")
    row1 = row1 / n1
    row2 = row2 - float(np.dot(row1, row2)) * row1
    n2 = float(np.linalg.norm(row2))
    if n2 <= 1e-12:
        raise ValueError("degenerate rot6d: parallel rows")
    row2 = row2 / n2
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert rot6d to quaternion (x,y,z,w)."""
    matrix = rot6d_to_matrix(rot6d)
    return Rotation.from_matrix(matrix).as_quat().astype(np.float64)


def eef_9d_to_end_pose_dict(eef_9d: np.ndarray) -> dict[str, Any]:
    """Decode absolute eef_9d (xyz + rot6d) into SDK set_end_pose JSON shape."""
    vec = np.asarray(eef_9d, dtype=np.float64).reshape(-1)
    if vec.shape != (9,):
        raise ValueError(f"expected eef_9d shape (9,), got {vec.shape}")
    quat = rot6d_to_quat_xyzw(vec[3:9])
    return {
        "position_m": {
            "x": float(vec[0]),
            "y": float(vec[1]),
            "z": float(vec[2]),
        },
        "orientation_xyzw": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }


def orientation_dict_to_rot6d(orientation: dict[str, Any]) -> list[float]:
    rot6d = quat_xyzw_to_rot6d(orientation_dict_to_quat_xyzw(orientation))
    return rot6d.astype(float).tolist()


def add_rot6d_to_end_pose(end_pose: dict[str, Any]) -> dict[str, Any]:
    """Return end_pose copy with rot6d field derived from orientation."""
    if "orientation" not in end_pose:
        raise KeyError("end_pose missing orientation")
    out = dict(end_pose)
    out["rot6d"] = orientation_dict_to_rot6d(end_pose["orientation"])
    return out
