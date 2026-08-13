#!/usr/bin/env python3
"""Step 4: export pipeline7 step3_rot6d -> LeRobot (state 32D / action 20D)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_export_utils import (
    ACTION_DIM,
    CAMERAS,
    CHUNK_SIZE,
    DEFAULT_MIN_DURATION_SEC_15FPS,
    FPS,
    MODALITY,
    STATE_DIM,
    compute_stats,
    episode_chunk,
    episode_passes_min_duration,
    extract_episode_arrays,
    fixed_size_float_array,
    infer_task_id,
    probe_video,
    validate_episode_vectors,
)
from manifest_utils import (
    CHECKPOINT_ROOT,
    DEFAULT_INPUT_ROOT,
    DEFAULT_MANIFEST,
    MULTI_DATASET,
    PIPELINE5_MULTI_DATASET,
    discover_episode_jsons_with_task_id,
    load_manifest,
    task_spec_map,
    write_json,
)

DEFAULT_OUTPUT_ROOT = MULTI_DATASET


def parse_task_ids(raw: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError("task-ids must not be empty")
    return parts


def build_task_index_map(
    task_ids: tuple[str, ...],
    *,
    export_mode: str,
) -> dict[str, int]:
    if export_mode == "single":
        if len(task_ids) != 1:
            raise ValueError("single export-mode requires exactly one task_id")
        return {task_ids[0]: 0}
    return {task_id: idx for idx, task_id in enumerate(task_ids)}


def filter_episodes_by_min_duration(
    episodes: list[tuple[Path, str]],
    *,
    min_duration_sec: float,
) -> tuple[list[tuple[Path, str]], list[dict[str, Any]]]:
    kept: list[tuple[Path, str]] = []
    rejected: list[dict[str, Any]] = []
    for input_json, task_id in episodes:
        ok, details = episode_passes_min_duration(input_json, min_duration_sec)
        if ok:
            kept.append((input_json, task_id))
        else:
            rejected.append(
                {
                    "source_episode_dir": str(input_json.parent),
                    "input_json": str(input_json),
                    "task_id": task_id,
                    "reason": "short_video_duration_step4",
                    **details,
                }
            )
    return kept, rejected


def ensure_chunk_dirs(output_root: Path, num_episodes: int) -> None:
    total_chunks = max(1, math.ceil(num_episodes / CHUNK_SIZE))
    for chunk_idx in range(total_chunks):
        (output_root / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        for original_key in CAMERAS.values():
            (output_root / "videos" / f"chunk-{chunk_idx:03d}" / original_key).mkdir(
                parents=True, exist_ok=True
            )


def export_episode(
    input_json: Path,
    episode_index: int,
    global_index_start: int,
    *,
    output_root: Path,
    task_index: int,
    task_text: str,
    link_videos_from: Path | None,
    use_symlink_videos: bool,
) -> dict[str, Any]:
    source_dir = input_json.parent
    task_id = infer_task_id(input_json)
    chunk_idx = episode_chunk(episode_index)

    with input_json.open(encoding="utf-8") as f:
        episode = json.load(f)

    arrays = extract_episode_arrays(episode)
    validation = validate_episode_vectors(arrays)
    n = arrays["num_frames"]

    table = pa.table(
        {
            "observation.state": fixed_size_float_array(arrays["state"]),
            "action": fixed_size_float_array(arrays["action"]),
            "timestamp": pa.array(arrays["timestamp_rel"], type=pa.float32()),
            "frame_index": pa.array(arrays["frame_index"], type=pa.int64()),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
            "index": pa.array(
                np.arange(global_index_start, global_index_start + n, dtype=np.int64),
                type=pa.int64(),
            ),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
        }
    )

    data_dir = output_root / "data" / f"chunk-{chunk_idx:03d}"
    video_root = output_root / "videos" / f"chunk-{chunk_idx:03d}"
    parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
    pq.write_table(table, parquet_path, compression="zstd", use_dictionary=False)

    video_checks: dict[str, Any] = {}
    for camera_key, original_key in CAMERAS.items():
        dst = video_root / original_key / f"episode_{episode_index:06d}.mp4"
        if link_videos_from is not None:
            src = (
                link_videos_from
                / "videos"
                / f"chunk-{chunk_idx:03d}"
                / original_key
                / f"episode_{episode_index:06d}.mp4"
            )
        else:
            src = source_dir / f"{camera_key}.mp4"

        if not src.is_file() and not src.is_symlink():
            raise FileNotFoundError(src)

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if link_videos_from is not None or use_symlink_videos:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        probe = probe_video(dst)
        video_checks[camera_key] = {
            **probe,
            "video_src": str(src.resolve()),
            "symlink": link_videos_from is not None or use_symlink_videos,
        }
        if probe["frame_count"] is not None and probe["frame_count"] != n:
            validation["issues"].append(
                f"{camera_key} frames {probe['frame_count']} != json {n}"
            )
            validation["ok"] = False

    return {
        "episode_index": episode_index,
        "chunk_index": chunk_idx,
        "task_id": task_id,
        "task_index": task_index,
        "task_text": task_text,
        "source_episode_dir": str(source_dir),
        "parquet_path": str(parquet_path),
        "length": n,
        "validation": validation,
        "videos": video_checks,
    }


def build_meta(
    episode_records: list[dict[str, Any]],
    total_frames: int,
    video_features: dict[str, Any],
    meta_dir: Path,
    *,
    task_entries: list[tuple[int, str]],
) -> None:
    total_episodes = len(episode_records)
    total_chunks = max(1, math.ceil(total_episodes / CHUNK_SIZE))

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
        "action": {"dtype": "float32", "shape": [ACTION_DIM]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    features.update(video_features)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "quanta_x1_biman_32d_state_20d_action_eef_only",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_entries),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": total_chunks,
        "total_videos": total_episodes * len(CAMERAS),
    }

    modality = {
        "state": MODALITY["state"],
        "action": MODALITY["action"],
        "video": {short: {"original_key": original} for short, original in CAMERAS.items()},
        "annotation": {
            "language.language_instruction": {"original_key": "task_index"},
        },
    }

    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "modality.json").write_text(
        json.dumps(modality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task_index, task_text in sorted(task_entries, key=lambda x: x[0]):
            f.write(
                json.dumps({"task_index": task_index, "task": task_text}, ensure_ascii=False)
                + "\n"
            )

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in episode_records:
            f.write(
                json.dumps(
                    {
                        "episode_index": rec["episode_index"],
                        "tasks": [rec["task_text"]],
                        "length": rec["length"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 step4: export LeRobot (32/20)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--export-mode", choices=("single", "multi"), default="multi")
    parser.add_argument("--task-ids", type=str, default="3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--link-videos-from",
        type=Path,
        default=PIPELINE5_MULTI_DATASET,
        help="Symlink videos from existing pipeline5 LeRobot export (same episode_index).",
    )
    parser.add_argument(
        "--no-link-videos-from",
        action="store_true",
        help="Copy/symlink mp4 from step3 input instead of pipeline5 lerobot.",
    )
    parser.add_argument(
        "--symlink-videos",
        action="store_true",
        help="When --no-link-videos-from, symlink step3 mp4 instead of copy.",
    )
    parser.add_argument("--min-duration-sec", type=float, default=DEFAULT_MIN_DURATION_SEC_15FPS)
    parser.add_argument("--no-clean-output", action="store_true")
    args = parser.parse_args()

    link_videos_from = None if args.no_link_videos_from else args.link_videos_from
    if link_videos_from is not None and not link_videos_from.is_dir():
        raise SystemExit(f"--link-videos-from not found: {link_videos_from}")

    manifest = load_manifest(args.manifest)
    specs = task_spec_map(args.manifest)
    task_ids = parse_task_ids(args.task_ids)
    task_index_map = build_task_index_map(task_ids, export_mode=args.export_mode)

    for task_id in task_ids:
        if task_id not in specs:
            raise SystemExit(f"task_id {task_id!r} not in manifest")
        if not specs[task_id].include:
            raise SystemExit(f"task_id {task_id!r} has include=false in manifest")

    task_text_map = {tid: specs[tid].text for tid in task_ids}
    task_entries = [(task_index_map[tid], task_text_map[tid]) for tid in task_ids]

    discovered = discover_episode_jsons_with_task_id(
        args.input_root,
        included_task_ids=set(task_ids),
    )
    if not discovered:
        raise SystemExit(f"No episodes under {args.input_root} for task_ids={task_ids}")

    rejected_duration: list[dict[str, Any]] = []
    discovered_total = len(discovered)
    pipeline_started = time.time()
    if args.min_duration_sec > 0:
        print(f"Step4 duration复核: min_video_duration >= {args.min_duration_sec}s @15fps")
        filter_started = time.time()
        discovered, rejected_duration = filter_episodes_by_min_duration(
            discovered,
            min_duration_sec=args.min_duration_sec,
        )
        print(
            f"  kept {len(discovered)}/{discovered_total} "
            f"({time.time() - pipeline_started:.1f}s)"
        )
        if not discovered:
            raise SystemExit("No episodes passed Step4 duration filter")

    if args.limit > 0:
        discovered = discovered[: args.limit]

    if args.output_root.exists() and not args.no_clean_output:
        shutil.rmtree(args.output_root)
    ensure_chunk_dirs(args.output_root, len(discovered))

    started = time.time()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    global_index = 0
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    video_features: dict[str, Any] = {}
    by_task: dict[str, int] = {}

    for episode_index, (input_json, task_id) in enumerate(discovered):
        try:
            rec = export_episode(
                input_json,
                episode_index,
                global_index,
                output_root=args.output_root,
                task_index=task_index_map[task_id],
                task_text=task_text_map[task_id],
                link_videos_from=link_videos_from,
                use_symlink_videos=args.symlink_videos,
            )
            records.append(rec)
            by_task[task_id] = by_task.get(task_id, 0) + 1
            if not rec["validation"]["ok"]:
                failures.append(rec)

            with input_json.open(encoding="utf-8") as f:
                episode = json.load(f)
            arrays = extract_episode_arrays(episode)
            all_states.append(arrays["state"])
            all_actions.append(arrays["action"])

            if not video_features:
                for camera_key, original_key in CAMERAS.items():
                    probe = rec["videos"][camera_key]
                    video_features[original_key] = {
                        "dtype": "video",
                        "shape": [probe["height"], probe["width"], 3],
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.height": probe["height"],
                            "video.width": probe["width"],
                            "video.codec": probe["codec"],
                            "video.pix_fmt": probe["pix_fmt"],
                            "video.is_depth_map": False,
                            "video.fps": probe["fps"],
                            "video.channels": 3,
                            "has_audio": False,
                        },
                    }

            global_index += rec["length"]
        except Exception as exc:
            failures.append({"input_json": str(input_json), "task_id": task_id, "error": repr(exc)})

        if (episode_index + 1) % 20 == 0 or episode_index + 1 == len(discovered):
            print(f"exported {episode_index + 1}/{len(discovered)} ({time.time() - started:.1f}s)")

    if not records:
        raise SystemExit("No episodes exported")

    meta_dir = args.output_root / "meta"
    build_meta(records, global_index, video_features, meta_dir, task_entries=task_entries)

    train_state = np.concatenate(all_states, axis=0)
    train_action = np.concatenate(all_actions, axis=0)
    stats = {
        "observation.state": compute_stats(train_state),
        "action": compute_stats(train_action),
    }
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "step": "step4_export_lerobot",
        "export_mode": args.export_mode,
        "task_ids": list(task_ids),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "link_videos_from": str(link_videos_from.resolve()) if link_videos_from else None,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "discovered_episodes": len(discovered) + len(rejected_duration),
        "rejected_duration_step4": rejected_duration,
        "exported_episodes": len(records),
        "exported_by_task_id": by_task,
        "total_frames": global_index,
        "validation_ok": sum(1 for r in records if r["validation"]["ok"]),
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(args.output_root / "export_report.json", report)

    print("\nPipeline7 step4 export finished.")
    print(f"  output : {args.output_root}")
    print(f"  state/action dim: {STATE_DIM}/{ACTION_DIM}")
    print(f"  episodes: {len(records)}/{len(discovered)}")
    print(f"  frames  : {global_index}")
    print(f"  by task : {by_task}")
    print(f"  validation ok: {report['validation_ok']}")
    if link_videos_from:
        print(f"  videos  : symlink from {link_videos_from}")

    if failures:
        print("\nFirst failures:")
        for f in failures[:5]:
            print(f"  {f}")

    if not records:
        write_json(
            args.output_root / "export_report.json",
            {
                "pipeline": "pipeline7_dongguan_relative_eef_only",
                "step": "step4_export_lerobot",
                "exported_episodes": 0,
                "failures": failures[:20],
                "failure_count": len(failures),
            },
        )
        raise SystemExit("No episodes exported; see export_report.json failures")


if __name__ == "__main__":
    main()
