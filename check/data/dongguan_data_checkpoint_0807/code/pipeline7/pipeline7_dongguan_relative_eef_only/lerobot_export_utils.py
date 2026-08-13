"""LeRobot export helpers for pipeline7: state 32D, action 20D (eef+gripper only)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

FPS = 15.0
CHUNK_SIZE = 1000
STATE_DIM = 32
ACTION_DIM = 20

STATE_MODALITY: dict[str, dict[str, int]] = {
    "left_eef_9d": {"start": 0, "end": 9},
    "left_gripper_position": {"start": 9, "end": 10},
    "left_joint_position": {"start": 10, "end": 16},
    "right_eef_9d": {"start": 16, "end": 25},
    "right_gripper_position": {"start": 25, "end": 26},
    "right_joint_position": {"start": 26, "end": 32},
}

ACTION_MODALITY: dict[str, dict[str, int]] = {
    "left_eef_9d": {"start": 0, "end": 9},
    "left_gripper_position": {"start": 9, "end": 10},
    "right_eef_9d": {"start": 10, "end": 19},
    "right_gripper_position": {"start": 19, "end": 20},
}

MODALITY: dict[str, dict[str, dict[str, int]]] = {
    "state": STATE_MODALITY,
    "action": ACTION_MODALITY,
}

CAMERAS = {
    "head_camera": "observation.images.head_camera",
    "left_arm_camera": "observation.images.left_arm_camera",
    "right_arm_camera": "observation.images.right_arm_camera",
}

DEFAULT_MIN_DURATION_SEC_15FPS = 3.5


def infer_task_id(episode_json: Path) -> str:
    from manifest_utils import batch_task_id

    batch_name = episode_json.parent.parent.name
    task_id = batch_task_id(batch_name)
    if task_id is None:
        raise ValueError(f"Cannot infer task_id from batch {batch_name!r}")
    return task_id


def episode_chunk(episode_index: int, *, chunk_size: int = CHUNK_SIZE) -> int:
    return episode_index // chunk_size


def end_pose_to_eef_9d(end_pose: dict[str, Any]) -> np.ndarray:
    pos = end_pose["position"]
    xyz = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
    rot6d = np.asarray(end_pose["rot6d"], dtype=np.float32)
    if rot6d.shape != (6,):
        raise ValueError(f"expected rot6d shape (6,), got {rot6d.shape}")
    return np.concatenate([xyz, rot6d])


def arm_state_vector(obs: dict[str, Any], *, side: str) -> np.ndarray:
    return np.concatenate(
        [
            end_pose_to_eef_9d(obs[f"{side}_arm_end_pose"]),
            np.asarray(obs[f"{side}_gripper_joint_states"]["positions"], dtype=np.float32),
            np.asarray(obs[f"{side}_arm_joint_states"]["positions"], dtype=np.float32),
        ]
    ).astype(np.float32)


def arm_action_vector_eef_only(act: dict[str, Any], *, side: str) -> np.ndarray:
    """Action slice: eef_9d + gripper only (no joint)."""
    return np.concatenate(
        [
            end_pose_to_eef_9d(act[f"{side}_arm_end_pose_action"]),
            np.asarray(act[f"{side}_gripper_actions"]["positions"], dtype=np.float32),
        ]
    ).astype(np.float32)


def frame_to_state_vector(frame: dict[str, Any]) -> np.ndarray:
    obs = frame["observation"]
    return np.concatenate(
        [arm_state_vector(obs, side="left"), arm_state_vector(obs, side="right")]
    ).astype(np.float32)


def frame_to_action_vector(frame: dict[str, Any]) -> np.ndarray:
    act = frame["action"]
    return np.concatenate(
        [
            arm_action_vector_eef_only(act, side="left"),
            arm_action_vector_eef_only(act, side="right"),
        ]
    ).astype(np.float32)


def state_vector_to_action_reference(state_vec: np.ndarray) -> np.ndarray:
    """Map 32D state → 20D action keys (eef+gripper slices)."""
    state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if state_vec.shape[0] != STATE_DIM:
        raise ValueError(f"expected state dim {STATE_DIM}, got {state_vec.shape}")
    return np.concatenate([state_vec[0:10], state_vec[16:26]]).astype(np.float32)


def extract_episode_arrays(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    frames = episode["frames"]
    n = len(frames)
    if n == 0:
        raise ValueError("episode has no frames")

    state = np.stack([frame_to_state_vector(fr) for fr in frames], axis=0)
    action = np.stack([frame_to_action_vector(fr) for fr in frames], axis=0)

    timestamps = np.array([float(fr["timestamp"]) for fr in frames], dtype=np.float64)
    timestamps_rel = (timestamps - timestamps[0]).astype(np.float32)
    frame_index = np.arange(n, dtype=np.int64)

    return {
        "state": state,
        "action": action,
        "timestamp_abs": timestamps,
        "timestamp_rel": timestamps_rel,
        "frame_index": frame_index,
        "num_frames": n,
    }


def validate_episode_vectors(arrays: dict[str, np.ndarray], *, tol: float = 1e-5) -> dict[str, Any]:
    state = arrays["state"]
    action = arrays["action"]
    n = len(state)
    issues: list[str] = []

    if state.shape[1] != STATE_DIM:
        issues.append(f"state dim {state.shape[1]} != {STATE_DIM}")
    if action.shape[1] != ACTION_DIM:
        issues.append(f"action dim {action.shape[1]} != {ACTION_DIM}")

    pairs = max(0, n - 1)
    full_match = 0
    for t in range(pairs):
        ref = state_vector_to_action_reference(state[t + 1])
        if np.allclose(action[t], ref, atol=tol, rtol=0.0):
            full_match += 1

    if pairs and full_match != pairs:
        issues.append(f"action[t]!=state[t+1] (20D slice): {full_match}/{pairs}")

    last_hold = bool(np.allclose(state_vector_to_action_reference(state[-1]), action[-1], atol=tol, rtol=0.0)) if n else True
    if n and not last_hold:
        issues.append("last frame hold-last failed")

    return {
        "num_frames": n,
        "checked_pairs": pairs,
        "full_vector_matches": full_match,
        "last_frame_hold_last": last_hold,
        "issues": issues,
        "ok": not issues,
    }


def fixed_size_float_array(matrix: np.ndarray) -> pa.Array:
    matrix = np.asarray(matrix, dtype=np.float32)
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, matrix.shape[1])


def compute_stats(matrix: np.ndarray) -> dict[str, list[float]]:
    matrix = np.asarray(matrix, dtype=np.float32)
    return {
        "mean": np.mean(matrix, axis=0).astype(np.float32).tolist(),
        "std": np.std(matrix, axis=0).astype(np.float32).tolist(),
        "min": np.min(matrix, axis=0).astype(np.float32).tolist(),
        "max": np.max(matrix, axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(matrix, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).astype(np.float32).tolist(),
    }


def _parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else None
    return float(value)


def probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    stream = json.loads(proc.stdout)["streams"][0]
    fps = _parse_fraction(stream.get("avg_frame_rate")) or FPS
    frame_count = stream.get("nb_frames")
    if frame_count in (None, "N/A", ""):
        frame_count = None
    else:
        frame_count = int(frame_count)
    duration_sec = float(stream["duration"]) if stream.get("duration") else None
    if duration_sec is None and frame_count is not None and fps:
        duration_sec = frame_count / fps
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": str(stream.get("codec_name", "")),
        "pix_fmt": str(stream.get("pix_fmt", "")),
        "fps": round(float(fps), 6),
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def episode_min_video_duration_sec(episode_json: Path) -> tuple[float | None, dict[str, float | None]]:
    source_dir = episode_json.parent
    durations = {
        camera_key: probe_video(source_dir / f"{camera_key}.mp4").get("duration_sec")
        for camera_key in CAMERAS
    }
    known = [dur for dur in durations.values() if dur is not None]
    return (min(known) if known else None), durations


def episode_passes_min_duration(
    episode_json: Path,
    min_duration_sec: float,
) -> tuple[bool, dict[str, Any]]:
    min_video_duration_sec, video_durations_sec = episode_min_video_duration_sec(episode_json)
    ok = min_video_duration_sec is not None and min_video_duration_sec >= min_duration_sec
    return ok, {
        "min_video_duration_sec": min_video_duration_sec,
        "video_durations_sec": video_durations_sec,
        "min_duration_sec": min_duration_sec,
    }
