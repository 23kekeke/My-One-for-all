"""Extract 32D bimanual state/action vectors and LeRobot metadata helpers."""

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
ARM_DIM = 16

# Canonical task strings (with trailing period) keyed by task folder name.
TASKS_BY_GROUP: dict[str, str] = {
    "task1": (
        "Extend only your right arm forward and press the button below the handle. "
        "Keep your left arm still."
    ),
    "task2": (
        "Use only your right hand to rotate the handle and open the door. "
        "Keep your left arm still."
    ),
    "task3": "Raise the left hand, and push the door to 90 degrees with the right hand.",
}

TASK_INDEX_BY_GROUP: dict[str, int] = {
    "task1": 0,
    "task2": 1,
    "task3": 2,
}

# 32D layout: left 16D + right 16D; each arm = eef_9d + gripper + joint(6).
MODALITY: dict[str, dict[str, dict[str, int]]] = {
    "state": {
        "left_eef_9d": {"start": 0, "end": 9},
        "left_gripper_position": {"start": 9, "end": 10},
        "left_joint_position": {"start": 10, "end": 16},
        "right_eef_9d": {"start": 16, "end": 25},
        "right_gripper_position": {"start": 25, "end": 26},
        "right_joint_position": {"start": 26, "end": 32},
    },
    "action": {
        "left_eef_9d": {"start": 0, "end": 9},
        "left_gripper_position": {"start": 9, "end": 10},
        "left_joint_position": {"start": 10, "end": 16},
        "right_eef_9d": {"start": 16, "end": 25},
        "right_gripper_position": {"start": 25, "end": 26},
        "right_joint_position": {"start": 26, "end": 32},
    },
}

CAMERAS = {
    "head_camera": "observation.images.head_camera",
    "left_arm_camera": "observation.images.left_arm_camera",
    "right_arm_camera": "observation.images.right_arm_camera",
}

DEFAULT_MIN_DURATION_SEC = 11.0


def infer_task_group(episode_json: Path) -> str:
    """Return task1/task2/task3 from path .../taskN/task_*_*/episode_*/episode.json."""
    for part in episode_json.parts:
        if part in TASKS_BY_GROUP:
            return part
    raise ValueError(f"Cannot infer task group from path: {episode_json}")


def canonical_task_for_group(task_group: str) -> str:
    if task_group not in TASKS_BY_GROUP:
        raise KeyError(f"Unknown task group: {task_group}")
    return TASKS_BY_GROUP[task_group]


def task_index_for_group(task_group: str) -> int:
    if task_group not in TASK_INDEX_BY_GROUP:
        raise KeyError(f"Unknown task group: {task_group}")
    return TASK_INDEX_BY_GROUP[task_group]


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


def arm_action_vector(act: dict[str, Any], *, side: str) -> np.ndarray:
    return np.concatenate(
        [
            end_pose_to_eef_9d(act[f"{side}_arm_end_pose_action"]),
            np.asarray(act[f"{side}_gripper_actions"]["positions"], dtype=np.float32),
            np.asarray(act[f"{side}_arm_actions"]["positions"], dtype=np.float32),
        ]
    ).astype(np.float32)


def frame_to_state_vector(frame: dict[str, Any]) -> np.ndarray:
    obs = frame["observation"]
    return np.concatenate(
        [
            arm_state_vector(obs, side="left"),
            arm_state_vector(obs, side="right"),
        ]
    ).astype(np.float32)


def frame_to_action_vector(frame: dict[str, Any]) -> np.ndarray:
    act = frame["action"]
    return np.concatenate(
        [
            arm_action_vector(act, side="left"),
            arm_action_vector(act, side="right"),
        ]
    ).astype(np.float32)


def extract_episode_arrays(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    frames = episode["frames"]
    n = len(frames)
    if n == 0:
        raise ValueError("episode has no frames")

    state = np.stack([frame_to_state_vector(fr) for fr in frames], axis=0)
    action = np.stack([frame_to_action_vector(fr) for fr in frames], axis=0)

    if state.shape != (n, STATE_DIM):
        raise ValueError(f"state shape {state.shape}, expected ({n}, {STATE_DIM})")
    if action.shape != (n, STATE_DIM):
        raise ValueError(f"action shape {action.shape}, expected ({n}, {STATE_DIM})")

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


def slice_vector(vector: np.ndarray, span: dict[str, int]) -> np.ndarray:
    return vector[int(span["start"]) : int(span["end"])]


def validate_episode_vectors(arrays: dict[str, np.ndarray], *, tol: float = 1e-5) -> dict[str, Any]:
    state = arrays["state"]
    action = arrays["action"]
    n = len(state)
    issues: list[str] = []

    if state.shape[1] != STATE_DIM or action.shape[1] != STATE_DIM:
        issues.append(f"bad dims: state={state.shape}, action={action.shape}")

    for modality_name, spans in MODALITY.items():
        for name, span in spans.items():
            if span["end"] - span["start"] <= 0:
                issues.append(f"bad span for {modality_name}.{name}")

    pairs = max(0, n - 1)
    full_match = 0
    segment_matches: dict[str, int] = {key: 0 for key in MODALITY["state"]}

    for t in range(pairs):
        s_next = state[t + 1]
        a_t = action[t]
        if np.allclose(a_t, s_next, atol=tol, rtol=0.0):
            full_match += 1
        for key, span in MODALITY["action"].items():
            if np.allclose(
                slice_vector(a_t, span),
                slice_vector(s_next, MODALITY["state"][key]),
                atol=tol,
                rtol=0.0,
            ):
                segment_matches[key] += 1

    if pairs and full_match != pairs:
        issues.append(f"action[t]!=state[t+1]: {full_match}/{pairs}")
    for key, count in segment_matches.items():
        if pairs and count != pairs:
            issues.append(f"{key} mismatch: {count}/{pairs}")

    last_hold = bool(np.allclose(state[-1], action[-1], atol=tol, rtol=0.0)) if n else True
    if n and not last_hold:
        issues.append("last frame hold-last failed")

    return {
        "num_frames": n,
        "checked_pairs": pairs,
        "full_vector_matches": full_match,
        "segment_matches": segment_matches,
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
    """Return min camera duration (sec) and per-camera durations."""
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
    ok = (
        min_video_duration_sec is not None
        and min_video_duration_sec >= min_duration_sec
    )
    return ok, {
        "min_video_duration_sec": min_video_duration_sec,
        "video_durations_sec": video_durations_sec,
        "min_duration_sec": min_duration_sec,
    }
