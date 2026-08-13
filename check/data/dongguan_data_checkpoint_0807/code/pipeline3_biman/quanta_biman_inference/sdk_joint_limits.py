"""SDK joint limit helpers (xr_lerobot-safe; no GR00T imports)."""

from __future__ import annotations

from typing import Literal

import numpy as np

ArmName = Literal["left", "right"]

SDK_RIGHT_ARM_LOWER = np.array(
    [-2.792, 0.0, -3.14, -1.57, -1.4, -1.745],
    dtype=np.float32,
)
SDK_RIGHT_ARM_UPPER = np.array(
    [2.792, 3.44, 0.0, 1.57, 1.4, 1.745],
    dtype=np.float32,
)
SDK_LEFT_ARM_LOWER = SDK_RIGHT_ARM_LOWER.copy()
SDK_LEFT_ARM_UPPER = SDK_RIGHT_ARM_UPPER.copy()


def clip_joints_to_sdk(arm: ArmName, joints: np.ndarray) -> np.ndarray:
    lower = SDK_LEFT_ARM_LOWER if arm == "left" else SDK_RIGHT_ARM_LOWER
    upper = SDK_LEFT_ARM_UPPER if arm == "right" else SDK_RIGHT_ARM_UPPER
    return np.clip(
        np.asarray(joints, dtype=np.float32).reshape(6),
        lower,
        upper,
    ).astype(np.float32)
