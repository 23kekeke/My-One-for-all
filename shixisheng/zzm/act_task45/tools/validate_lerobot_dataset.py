#!/usr/bin/env python3
"""Load the converted dataset through LeRobot and verify the ACT contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REQUIRED_CAMERAS = (
    "observation.images.head_camera",
    "observation.images.left_arm_camera",
    "observation.images.right_arm_camera",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    features = dataset.meta.features

    for key in ("observation.state", "action", *REQUIRED_CAMERAS):
        if key not in features:
            raise SystemExit(f"Missing required feature: {key}")

    state_shape = tuple(features["observation.state"]["shape"])
    action_shape = tuple(features["action"]["shape"])
    if state_shape != (14,) or action_shape != (14,):
        raise SystemExit(f"Expected state/action shape (14,), got {state_shape}/{action_shape}")

    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in indices:
        sample = dataset[index]
        state = sample["observation.state"]
        action = sample["action"]
        if tuple(state.shape) != (14,) or tuple(action.shape) != (14,):
            raise SystemExit(
                f"Bad sample shape at index {index}: state={tuple(state.shape)}, action={tuple(action.shape)}"
            )
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise SystemExit(f"NaN or Inf in sample {index}")
        for camera in REQUIRED_CAMERAS:
            image = sample[camera]
            if image.ndim != 3 or image.shape[0] != 3:
                raise SystemExit(f"Bad image shape for {camera} at index {index}: {tuple(image.shape)}")

    print(f"root: {args.root.resolve()}")
    print(f"repo_id: {dataset.repo_id}")
    print(f"episodes: {dataset.num_episodes}")
    print(f"frames: {len(dataset)}")
    print(f"fps: {dataset.fps}")
    print(f"state_shape: {state_shape}")
    print(f"action_shape: {action_shape}")
    print(f"camera_keys: {dataset.meta.camera_keys}")
    print(f"decoded_sample_indices: {indices}")
    print("LEROBOT DATASET VALIDATION PASSED")


if __name__ == "__main__":
    main()

