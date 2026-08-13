"""Load task_manifest.yaml and discover dongguan episodes (pipeline7)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BATCH_RE = re.compile(r"^task_(\d+)_\d+$")
CHECKPOINT_ROOT = Path("/data/dongguan_data_checkpoint_0807")
PIPELINE5_CHECKPOINT_ROOT = CHECKPOINT_ROOT
DEFAULT_INPUT_ROOT = CHECKPOINT_ROOT / "tmp/step3_rot6d"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "task_manifest.yaml"
MULTI_DATASET = CHECKPOINT_ROOT / "lerobot_p7/multi_345"
PIPELINE5_MULTI_DATASET = CHECKPOINT_ROOT / "lerobot/multi_345"
SMOKE_ROOT = CHECKPOINT_ROOT / "tmp/step_smoke_p7"
OUTPUT_ROOT = CHECKPOINT_ROOT / "output_p7"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    text: str
    include: bool
    min_duration_sec: float
    max_duration_sec: float | None


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (expected mapping): {path}")
    return data


def parse_task_specs(manifest: dict[str, Any]) -> list[TaskSpec]:
    defaults = manifest.get("defaults") or {}
    default_min = float(defaults.get("min_duration_sec", 4.0))
    raw_tasks = manifest.get("tasks") or []
    specs: list[TaskSpec] = []
    seen: set[str] = set()
    for entry in raw_tasks:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid task entry: {entry!r}")
        task_id = str(entry["task_id"]).strip()
        if task_id in seen:
            raise ValueError(f"Duplicate task_id in manifest: {task_id!r}")
        seen.add(task_id)
        max_dur = entry.get("max_duration_sec")
        specs.append(
            TaskSpec(
                task_id=task_id,
                text=str(entry["text"]).strip(),
                include=bool(entry.get("include", True)),
                min_duration_sec=float(entry.get("min_duration_sec", default_min)),
                max_duration_sec=float(max_dur) if max_dur is not None else None,
            )
        )
    return specs


def included_task_specs(manifest_path: Path) -> list[TaskSpec]:
    specs = parse_task_specs(load_manifest(manifest_path))
    included = [s for s in specs if s.include]
    if not included:
        raise ValueError(f"No tasks with include=true in {manifest_path}")
    for spec in included:
        if not spec.text:
            raise ValueError(f"task_id={spec.task_id!r} has empty text")
    return included


def task_spec_map(manifest_path: Path) -> dict[str, TaskSpec]:
    return {s.task_id: s for s in parse_task_specs(load_manifest(manifest_path))}


def batch_task_id(batch_name: str) -> str | None:
    match = BATCH_RE.match(batch_name)
    return match.group(1) if match else None


def discover_episode_jsons_with_task_id(
    input_root: Path,
    *,
    included_task_ids: set[str],
) -> list[tuple[Path, str]]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    found: list[tuple[Path, str]] = []
    for batch_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        tid = batch_task_id(batch_dir.name)
        if tid is None or tid not in included_task_ids:
            continue
        for episode_json in sorted(batch_dir.glob("episode_*/episode.json")):
            found.append((episode_json, tid))
    return found


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
