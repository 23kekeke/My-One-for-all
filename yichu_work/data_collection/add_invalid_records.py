from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
import random
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

def load_metadata(path: str | Path) -> dict:
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("episodes"), list):
        raise ValueError(f"invalid metadata structure: {path}")
    invalid_count = metadata.get("invalid_count")
    if invalid_count is not None and invalid_count != len(metadata["episodes"]):
        raise ValueError(
            f"invalid_count mismatch: {invalid_count} != {len(metadata['episodes'])}"
        )
    return metadata


def atomic_write_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Episode timestamp: {value!r}") from error


def _record_key(record: dict) -> tuple:
    return record.get("episode_id"), record.get("timestamp"), record.get("path")


def _timedelta_microseconds(delta: timedelta) -> int:
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _allowed_offset(rank: int, maximum: int, forbidden: list[int]) -> int:
    """Map a zero-based rank to an offset while skipping forbidden offsets."""
    low, high = 1, maximum
    while low < high:
        middle = (low + high) // 2
        allowed_through_middle = middle - bisect_right(forbidden, middle)
        if allowed_through_middle > rank:
            high = middle
        else:
            low = middle + 1
    return low


def append_invalid_record(
    path: str | Path,
    dataset_metadata: dict,
    episode_summary: dict,
) -> dict:
    """Copy one valid Episode summary and add only ``invalid: true``."""
    path = Path(path)
    if path.exists():
        invalid_metadata = load_metadata(path)
    else:
        invalid_metadata = {
            key: deepcopy(value)
            for key, value in dataset_metadata.items()
            if key != "episodes"
        }
        invalid_metadata["episodes"] = []

    invalid_record = deepcopy(episode_summary)
    invalid_record["invalid"] = True
    key = _record_key(invalid_record)
    if not any(_record_key(record) == key for record in invalid_metadata["episodes"]):
        invalid_metadata["episodes"].append(invalid_record)
    invalid_metadata["episodes"].sort(key=lambda record: _parse_timestamp(record["timestamp"]))

    timestamps = [_parse_timestamp(record["timestamp"]) for record in invalid_metadata["episodes"]]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("invalid Episode timestamps must be strictly increasing")
    invalid_metadata["invalid_count"] = len(invalid_metadata["episodes"])

    atomic_write_metadata(path, invalid_metadata)
    written = load_metadata(path)
    if not any(_record_key(record) == key for record in written["episodes"]):
        raise RuntimeError("invalid Episode summary was not persisted")
    return invalid_record


def add_random_records(
    metadata: dict,
    valid_metadata: dict,
    count: int,
    rng=None,
) -> tuple[dict, list[str]]:
    """Generate exactly ``count`` unique timestamps across the valid time range."""
    if count < 0:
        raise ValueError("count must be non-negative")
    result = deepcopy(metadata)
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("at least one invalid Episode summary is required as a template")
    episodes.sort(key=lambda record: _parse_timestamp(record["timestamp"]))
    invalid_timestamps = [_parse_timestamp(record["timestamp"]) for record in episodes]
    if any(
        current <= previous
        for previous, current in zip(invalid_timestamps, invalid_timestamps[1:])
    ):
        raise ValueError("existing invalid Episode timestamps are not strictly increasing")

    valid_episodes = valid_metadata.get("episodes")
    if not isinstance(valid_episodes, list) or len(valid_episodes) < 2:
        raise ValueError("at least two valid Episode summaries are required")
    valid_timestamps = sorted(
        _parse_timestamp(record["timestamp"]) for record in valid_episodes
    )
    if any(
        current <= previous
        for previous, current in zip(valid_timestamps, valid_timestamps[1:])
    ):
        raise ValueError("valid Episode timestamps must be strictly increasing")

    start, end = valid_timestamps[0], valid_timestamps[-1]
    maximum_offset = _timedelta_microseconds(end - start) - 1
    if maximum_offset < 1:
        raise ValueError("valid Episode time range has no available microsecond positions")

    occupied_timestamps = set(valid_timestamps) | set(invalid_timestamps)
    forbidden_offsets = sorted(
        {
            _timedelta_microseconds(timestamp - start)
            for timestamp in occupied_timestamps
            if start < timestamp < end
        }
    )
    available_count = maximum_offset - len(forbidden_offsets)
    if count > available_count:
        raise ValueError(
            f"count {count} exceeds {available_count} available unique timestamps "
            "inside the valid Episode time range"
        )

    generator = rng or random.SystemRandom()
    template = deepcopy(episodes[-1])
    selected_ranks = generator.sample(range(available_count), count)
    selected_offsets = sorted(
        _allowed_offset(rank, maximum_offset, forbidden_offsets)
        for rank in selected_ranks
    )
    generated_timestamps = []
    for offset_us in selected_offsets:
        generated_time = start + timedelta(microseconds=offset_us)
        generated = deepcopy(template)
        generated["timestamp"] = generated_time.isoformat(timespec="microseconds")
        generated["invalid"] = True
        generated["generated"] = True
        episodes.append(generated)
        generated_timestamps.append(generated["timestamp"])
    episodes.sort(key=lambda record: _parse_timestamp(record["timestamp"]))
    result["invalid_count"] = len(episodes)
    return result, generated_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append random, strictly increasing invalid Episode summaries"
    )
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument(
        "--valid-file",
        type=Path,
        help="valid dataset_metadata.json (default: inferred from invalid metadata path)",
    )
    parser.add_argument("--count", required=True, type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    valid_file = args.valid_file or args.file.parent.parent / "dataset_metadata.json"
    original = load_metadata(args.file)
    valid_metadata = load_metadata(valid_file)
    updated, generated_timestamps = add_random_records(
        original, valid_metadata, args.count
    )
    timestamps = [record["timestamp"] for record in updated["episodes"]]
    print(f"FILE={args.file}")
    print(f"VALID_FILE={valid_file}")
    print(f"BEFORE={len(original['episodes'])}")
    print(f"ADDED={args.count}")
    print(f"AFTER={len(updated['episodes'])}")
    print(f"INVALID_COUNT={updated['invalid_count']}")
    print(f"FIRST_TIMESTAMP={timestamps[0]}")
    print(f"LAST_TIMESTAMP={timestamps[-1]}")
    valid_range = sorted(
        _parse_timestamp(record["timestamp"])
        for record in valid_metadata["episodes"]
    )
    print(
        f"VALID_TIME_RANGE={valid_range[0].isoformat(timespec='microseconds')}.."
        f"{valid_range[-1].isoformat(timespec='microseconds')}"
    )
    print(f"GENERATED_TIMESTAMPS={generated_timestamps}")
    if args.apply:
        atomic_write_metadata(args.file, updated)
        verified = load_metadata(args.file)
        if verified != updated:
            raise RuntimeError("written invalid metadata failed read-back verification")
        print("APPLIED=true")
    else:
        print("APPLIED=false")


if __name__ == "__main__":
    main()
