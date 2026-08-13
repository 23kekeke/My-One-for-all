#!/usr/bin/env python3
"""Check raw dataset structure without assuming a fixed episode count."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


REQUIRED_FILES = {
    "episode.json",
    "head_camera.mp4",
    "left_arm_camera.mp4",
    "right_arm_camera.mp4",
}


def print_names(label: str, names: set[str] | list[str], limit: int = 10) -> None:
    ordered = sorted(names)
    print(f"{label}: {len(ordered)}")
    if ordered:
        suffix = "" if len(ordered) <= limit else f" ... (+{len(ordered) - limit})"
        print(f"  {', '.join(ordered[:limit])}{suffix}")


def validate_episode_content(path: Path) -> str | None:
    try:
        episode = json.loads(path.read_text(encoding="utf-8"))
        frames = episode.get("frames")
        if not isinstance(frames, list) or not frames:
            return "frames is empty"

        previous_timestamp = None
        for index, frame in enumerate(frames):
            timestamp = float(frame["timestamp"])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                return f"timestamp is not increasing at frame {index}"
            previous_timestamp = timestamp

            observation = frame["observation"]
            action = frame["action"]
            state_values = (
                observation["left_arm_joint_states"]["positions"]
                + observation["left_gripper_joint_states"]["positions"]
                + observation["right_arm_joint_states"]["positions"]
                + observation["right_gripper_joint_states"]["positions"]
            )
            action_values = (
                action["left_arm_actions"]["positions"]
                + action["left_gripper_actions"]["positions"]
                + action["right_arm_actions"]["positions"]
                + action["right_gripper_actions"]["positions"]
            )
            if len(state_values) != 14 or len(action_values) != 14:
                return f"state/action dimension is {len(state_values)}/{len(action_values)} at frame {index}"
            if not all(math.isfinite(float(value)) for value in state_values + action_values):
                return f"NaN or Inf at frame {index}"
    except Exception as exc:
        return str(exc)
    return None


def check_dataset(root: Path, deep_json: bool, actual_only: bool) -> bool:
    print("\n" + "=" * 72)
    print(f"Dataset: {root}")

    actual = {path.name for path in root.glob("episode_*") if path.is_dir()}
    metadata_path = root / "dataset_metadata.json"

    metadata_warning = None
    if not metadata_path.is_file():
        if not actual_only:
            print(f"Actual episode directories: {len(actual)}")
            print("Status: FAILED (missing dataset_metadata.json)")
            return False
        metadata = {"episodes": []}
        metadata_warning = "missing dataset_metadata.json"

    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            if not actual_only:
                print(f"Actual episode directories: {len(actual)}")
                print(f"Status: FAILED (invalid dataset_metadata.json: {exc})")
                return False
            metadata = {"episodes": []}
            metadata_warning = f"invalid dataset_metadata.json: {exc}"

    entries = metadata.get("episodes", [])
    if not isinstance(entries, list):
        print("Status: FAILED (metadata field 'episodes' is not a list)")
        return False

    metadata_names = [
        str(entry.get("path", f"episode_{int(entry.get('episode_id', -1)):04d}")) for entry in entries
    ]
    metadata_set = set(metadata_names)
    duplicates = {name for name, count in Counter(metadata_names).items() if count > 1}
    missing_directories = metadata_set - actual
    orphan_directories = actual - metadata_set

    incomplete: list[str] = []
    invalid_episode_json: list[str] = []
    names_to_check = actual if actual_only else metadata_set & actual
    for name in sorted(names_to_check):
        episode_dir = root / name
        missing_files = sorted(file for file in REQUIRED_FILES if not (episode_dir / file).is_file())
        if missing_files:
            incomplete.append(f"{name}({','.join(missing_files)})")
            continue

        if deep_json:
            error = validate_episode_content(episode_dir / "episode.json")
            if error is not None:
                invalid_episode_json.append(f"{name}({error})")

    complete = len(names_to_check) - len(incomplete) - len(invalid_episode_json)
    if actual_only:
        passed = bool(actual) and not incomplete and not invalid_episode_json
    else:
        passed = not any(
            [duplicates, missing_directories, orphan_directories, incomplete, invalid_episode_json]
        )

    print(f"Metadata episodes: {len(entries)}")
    print(f"Unique metadata paths: {len(metadata_set)}")
    print(f"Actual episode directories: {len(actual)}")
    print(f"Complete episodes: {complete}")
    if actual_only:
        print("Count policy: actual episode directories (metadata count mismatch is warning only)")
    if metadata_warning:
        print(f"Metadata warning: {metadata_warning}")
    print_names("Missing episode directories", missing_directories)
    print_names("Orphan episode directories", orphan_directories)
    print_names("Duplicate metadata paths", duplicates)
    print_names("Incomplete episodes", incomplete)
    if deep_json:
        print_names("Invalid episode JSON", invalid_episode_json)
    print(f"Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, help="Check one dataset directory directly")
    parser.add_argument("--base", type=Path, help="Directory containing raw datasets")
    parser.add_argument("--pattern", default="*", help="Dataset directory glob, e.g. task_45_*")
    parser.add_argument(
        "--deep-json",
        action="store_true",
        help="Also parse every episode.json; slower but more thorough",
    )
    parser.add_argument(
        "--actual-only",
        action="store_true",
        help="Use actual episode directories; metadata count mismatch is warning only",
    )
    args = parser.parse_args()

    if args.dataset is not None and args.base is not None:
        parser.error("use either --dataset or --base, not both")
    if args.dataset is not None:
        roots = [args.dataset]
    elif args.base is not None:
        roots = sorted(path for path in args.base.glob(args.pattern) if path.is_dir())
    else:
        parser.error("one of --dataset or --base is required")
    if not roots:
        raise SystemExit("No dataset directories matched")

    results = [check_dataset(root, args.deep_json, args.actual_only) for root in roots]
    passed = sum(results)
    print("\n" + "=" * 72)
    print(f"Summary: {passed}/{len(results)} datasets PASSED")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
