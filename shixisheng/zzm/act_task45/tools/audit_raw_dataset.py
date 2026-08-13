#!/usr/bin/env python3
"""Read-only audit for the raw task_45 SDK dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def raw14(frame: dict, section: str) -> list[float]:
    data = frame[section]
    if section == "observation":
        return (
            data["left_arm_joint_states"]["positions"]
            + data["left_gripper_joint_states"]["positions"]
            + data["right_arm_joint_states"]["positions"]
            + data["right_gripper_joint_states"]["positions"]
        )
    return (
        data["left_arm_actions"]["positions"]
        + data["left_gripper_actions"]["positions"]
        + data["right_arm_actions"]["positions"]
        + data["right_gripper_actions"]["positions"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episodes = metadata.get("episodes", [])
    if not episodes:
        raise SystemExit("No episodes listed in dataset_metadata.json")

    missing: list[str] = []
    total_frames = 0
    total_duration = 0.0
    required_files = (
        "episode.json",
        "head_camera.mp4",
        "left_arm_camera.mp4",
        "right_arm_camera.mp4",
    )
    for episode in episodes:
        episode_dir = root / episode["path"]
        total_frames += int(episode["num_frames"])
        total_duration += float(episode["duration"])
        for name in required_files:
            if not (episode_dir / name).is_file():
                missing.append(str(episode_dir / name))

    first_path = root / episodes[0]["path"] / "episode.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    frames = first["frames"]
    if not frames:
        raise SystemExit(f"No frames in {first_path}")

    state0 = raw14(frames[0], "observation")
    action0 = raw14(frames[0], "action")
    if len(state0) != 14 or len(action0) != 14:
        raise SystemExit(f"Expected state/action dimension 14, got {len(state0)}/{len(action0)}")

    abs_diffs: list[float] = []
    frame_dts: list[float] = []
    for index, frame in enumerate(frames):
        state = raw14(frame, "observation")
        action = raw14(frame, "action")
        if len(state) != 14 or len(action) != 14:
            raise SystemExit(f"Invalid dimension at frame {index}: {len(state)}/{len(action)}")
        if not all(math.isfinite(float(value)) for value in state + action):
            raise SystemExit(f"NaN or Inf at frame {index}")
        abs_diffs.extend(abs(float(a) - float(s)) for a, s in zip(action, state, strict=True))
        if index:
            frame_dts.append(float(frame["timestamp"]) - float(frames[index - 1]["timestamp"]))

    print(f"root: {root}")
    print(f"episodes: {len(episodes)}")
    print(f"frames: {total_frames}")
    print(f"duration_minutes: {total_duration / 60:.2f}")
    print(f"fps_metadata: {metadata.get('fps')}")
    print(f"first_episode_frames: {len(frames)}")
    print(f"state_dim: {len(state0)}")
    print(f"action_dim: {len(action0)}")
    print(f"first_episode_mean_dt_s: {sum(frame_dts) / len(frame_dts):.6f}")
    print(f"first_episode_mean_abs_action_minus_state: {sum(abs_diffs) / len(abs_diffs):.9f}")
    print(f"missing_required_files: {len(missing)}")
    for path in missing[:20]:
        print(f"  missing: {path}")

    if missing:
        raise SystemExit("Raw dataset audit failed: required files are missing")
    print("RAW DATASET AUDIT PASSED")


if __name__ == "__main__":
    main()

