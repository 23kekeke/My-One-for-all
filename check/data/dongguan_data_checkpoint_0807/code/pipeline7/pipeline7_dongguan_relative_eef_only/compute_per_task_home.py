#!/usr/bin/env python3
"""Compute per-task deploy home poses from LeRobot training data (frame 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PIPELINE7 = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE7))

from lerobot_export_utils import STATE_MODALITY
from manifest_utils import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_MANIFEST,
    MULTI_DATASET,
    SMOKE_ROOT,
    discover_episode_jsons_with_task_id,
    included_task_specs,
    write_json,
)


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    r1 = rot6d[:3]
    r2 = rot6d[3:]
    r1 = r1 / max(np.linalg.norm(r1), 1e-12)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / max(np.linalg.norm(r2), 1e-12)
    r3 = np.cross(r1, r2)
    matrix = np.stack([r1, r2, r3], axis=0)
    return Rotation.from_matrix(matrix).as_quat()


def slice_state(state: np.ndarray, key: str) -> np.ndarray:
    span = STATE_MODALITY[key]
    return np.asarray(state, dtype=np.float64)[span["start"] : span["end"]]


def end_pose_from_eef(eef_9d: np.ndarray) -> dict[str, Any]:
    eef_9d = np.asarray(eef_9d, dtype=np.float64)
    quat = rot6d_to_quat_xyzw(eef_9d[3:])
    return {
        "position": {
            "x": float(eef_9d[0]),
            "y": float(eef_9d[1]),
            "z": float(eef_9d[2]),
        },
        "orientation_xyzw": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }


def summarize_group(states: np.ndarray) -> dict[str, Any]:
    """states: (N, D)"""
    median = np.median(states, axis=0)
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)
    q10 = np.quantile(states, 0.10, axis=0)
    q90 = np.quantile(states, 0.90, axis=0)
    return {
        "count": int(states.shape[0]),
        "median": median.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "q10": q10.astype(float).tolist(),
        "q90": q90.astype(float).tolist(),
    }


def arm_end_pose_home(eef_9d: np.ndarray, gripper: float) -> dict[str, Any]:
    """Deploy home: SDK set_end_pose (xyz + quat xyzw) + gripper only."""
    pose = end_pose_from_eef(eef_9d)
    return {
        "gripper_position": gripper,
        "end_pose": {
            "position_m": pose["position"],
            "orientation_xyzw": pose["orientation_xyzw"],
        },
    }


def build_task_home_entry(
    *,
    task_index: int,
    task_id: str,
    language: str,
    episode_count: int,
    median_state: np.ndarray,
    lift_position_m: float | None,
) -> dict[str, Any]:
    left_eef = slice_state(median_state, "left_eef_9d")
    right_eef = slice_state(median_state, "right_eef_9d")
    left_grip = float(slice_state(median_state, "left_gripper_position")[0])
    right_grip = float(slice_state(median_state, "right_gripper_position")[0])
    entry: dict[str, Any] = {
        "task_index": task_index,
        "task_id": task_id,
        "language": language,
        "episode_count": episode_count,
        "execute_arms": ["right"],
        "left_arm": arm_end_pose_home(left_eef, left_grip),
        "right_arm": arm_end_pose_home(right_eef, right_grip),
    }
    if lift_position_m is not None:
        entry["lift_position_m"] = lift_position_m
    return entry


def load_frame0_states(dataset_path: Path) -> tuple[list[np.ndarray], list[int]]:
    data_dir = dataset_path / "data/chunk-000"
    paths = sorted(data_dir.glob("episode_*.parquet"))
    states: list[np.ndarray] = []
    task_indices: list[int] = []
    for path in paths:
        table = pq.read_table(path, columns=["observation.state", "task_index"])
        row = table.slice(0, 1).to_pydict()
        states.append(np.asarray(row["observation.state"][0], dtype=np.float64))
        task_indices.append(int(row["task_index"][0]))
    return states, task_indices


def load_lift_frame0(
    *,
    manifest_path: Path,
    input_root: Path,
    task_ids: list[str],
) -> dict[str, list[float]]:
    specs = included_task_specs(manifest_path)
    included_ids = {s.task_id for s in specs if s.include}
    discovered = discover_episode_jsons_with_task_id(
        input_root,
        included_task_ids=included_ids,
    )
    lifts_by_task: dict[str, list[float]] = {tid: [] for tid in task_ids}
    for input_json, task_id in discovered:
        if task_id not in lifts_by_task:
            continue
        episode = json.loads(input_json.read_text(encoding="utf-8"))
        frames = episode.get("frames") or []
        if not frames:
            continue
        lift = frames[0]["observation"].get("lift_joint_states", {}).get("positions", [None])[0]
        if lift is not None:
            lifts_by_task[task_id].append(float(lift))
    return lifts_by_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-task deploy home from training data.")
    parser.add_argument("--dataset-path", type=Path, default=MULTI_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PIPELINE7 / "deploy" / "per_task_home.json",
        help="DGX deploy JSON (primary output).",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=SMOKE_ROOT / "per_task_home_report.json",
        help="Optional detailed stats report with spread.",
    )
    args = parser.parse_args()

    tasks_path = args.dataset_path / "meta/tasks.jsonl"
    task_map: dict[int, dict[str, str]] = {}
    with tasks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                task_map[int(row["task_index"])] = {
                    "task_index": str(row["task_index"]),
                    "language": row["task"],
                }

    task_id_by_index = {"0": "3", "1": "4", "2": "5"}

    states, task_indices = load_frame0_states(args.dataset_path)
    grouped: dict[int, list[np.ndarray]] = {}
    for state, task_index in zip(states, task_indices):
        grouped.setdefault(task_index, []).append(state)

    lifts_by_task = load_lift_frame0(
        manifest_path=args.manifest,
        input_root=args.input_root,
        task_ids=["3", "4", "5"],
    )

    per_task_home: list[dict[str, Any]] = []
    report_tasks: dict[str, Any] = {}
    for task_index in sorted(grouped):
        arr = np.stack(grouped[task_index], axis=0)
        summary = summarize_group(arr)
        median_state = np.asarray(summary["median"], dtype=np.float64)

        task_id = task_id_by_index.get(str(task_index), str(task_index))
        lift_vals = lifts_by_task.get(task_id, [])
        lift_m = float(np.median(lift_vals)) if lift_vals else None

        entry = build_task_home_entry(
            task_index=task_index,
            task_id=task_id,
            language=task_map[task_index]["language"],
            episode_count=summary["count"],
            median_state=median_state,
            lift_position_m=lift_m,
        )
        per_task_home.append(entry)
        report_tasks[str(task_index)] = {
            **entry,
            "spread_state_32d": {
                "std": summary["std"],
                "q10": summary["q10"],
                "q90": summary["q90"],
            },
            "lift_spread": {
                "count": len(lift_vals),
                "median": lift_m,
                "min": float(np.min(lift_vals)) if lift_vals else None,
                "max": float(np.max(lift_vals)) if lift_vals else None,
                "std": float(np.std(lift_vals)) if lift_vals else None,
            },
        }

    deploy_json = {
        "schema_version": "2",
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "dataset_path": str(args.dataset_path.resolve()),
        "stats_method": "median(observation.state[t=0]) per episode",
        "total_episodes": len(states),
        "preposition_mode": "end_pose",
        "orientation_format": "quaternion_xyzw",
        "orientation_note": (
            "SDK set_end_pose uses position_m (xyz) + orientation_xyzw (4D). "
            "Do not send rot6d to SDK; rot6d is training-only inside GR00T eef_9d."
        ),
        "coordinate_note": (
            "end_pose xyz in SDK arm-base frame (not world); "
            "lift_position_m is separate LiftController target"
        ),
        "per_task_home": per_task_home,
    }

    write_json(args.output, deploy_json)
    write_json(
        args.report_output,
        {
            "pipeline": "pipeline7_dongguan_relative_eef_only",
            "step": "compute_per_task_home",
            "deploy_json": str(args.output.resolve()),
            "source_input_root": str(args.input_root.resolve()),
            "tasks": report_tasks,
        },
    )

    print("Per-task home computed:")
    for row in per_task_home:
        pos = row["right_arm"]["end_pose"]["position_m"]
        print(
            f"  task_index={row['task_index']} task_id={row['task_id']} "
            f"n={row['episode_count']} "
            f"right_xyz=[{pos['x']:.3f},{pos['y']:.3f},{pos['z']:.3f}] "
            f"lift={row.get('lift_position_m')}"
        )
    print(f"  DGX json: {args.output}")
    print(f"  stats report: {args.report_output}")


if __name__ == "__main__":
    main()
