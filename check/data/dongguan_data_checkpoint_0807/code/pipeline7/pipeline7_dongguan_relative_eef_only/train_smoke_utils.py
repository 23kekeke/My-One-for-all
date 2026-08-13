"""Shared helpers for pipeline7 Step 7 overfit smoke."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from manifest_utils import MULTI_DATASET, OUTPUT_ROOT, SMOKE_ROOT

ROOT = Path(__file__).resolve().parents[1]
GR00T_REPO = ROOT / "isaacGr00t"
PIPELINE2 = ROOT / "pipeline2"
PIPELINE3 = ROOT / "pipeline3_biman"
PIPELINE7 = Path(__file__).resolve().parent

BASE_MODEL = ROOT / "GR00T-N1.7-3B"
EP0_DATASET = SMOKE_ROOT / "multi_345_ep0"
OVERFIT_OUTPUT = OUTPUT_ROOT / "overfit_ep0_smoke"
GR00T_PYTHON = GR00T_REPO / ".venv/bin/python"
RELATIVE_MODALITY_CONFIG = PIPELINE7 / "dongguan_eef_only_relative_modality.py"
RELATIVE_FINETUNE_ENTRY = PIPELINE7 / "dongguan_eef_only_finetune_entry.py"
NUM_CAMERAS = 3


def setup_gr00t_env() -> dict[str, str]:
    env = os.environ.copy()
    pipeline_paths = [str(PIPELINE7), str(GR00T_REPO), str(PIPELINE2), str(PIPELINE3)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(pipeline_paths + ([existing] if existing else []))
    env["GR00T_COSMOS_MODEL_PATH"] = str(ROOT / "Cosmos-Reason2-2B")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["LOGURU_LEVEL"] = env.get("LOGURU_LEVEL", "INFO")
    env["GROOT_PATCH_MISTRAL"] = "1"
    env["GROOT_HF_LOCAL_FIRST"] = "1"
    env["PIPELINE2_QLORA"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    return env


def load_episodes(meta_dir: Path) -> list[dict[str, Any]]:
    with (meta_dir / "episodes.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def symlink_if_needed(link_path: Path, target: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Expected symlink path but found: {link_path}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target.resolve(), link_path)


def create_single_episode_view(
    episode_index: int,
    *,
    source_root: Path = MULTI_DATASET,
    out_root: Path = EP0_DATASET,
) -> dict[str, Any]:
    episodes = load_episodes(source_root / "meta")
    lookup = {int(ep["episode_index"]): ep for ep in episodes}
    if episode_index not in lookup:
        raise KeyError(f"episode_index {episode_index} not found in {source_root}")

    if out_root.exists():
        shutil.rmtree(out_root)
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True)

    symlink_if_needed(out_root / "data", source_root / "data")
    symlink_if_needed(out_root / "videos", source_root / "videos")

    for name in ("modality.json", "tasks.jsonl", "stats.json", "relative_stats.json"):
        src = source_root / "meta" / name
        if src.is_file():
            shutil.copy2(src, meta_dir / name)

    ep_meta = lookup[episode_index]
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(ep_meta, ensure_ascii=False) + "\n")

    with (source_root / "meta/info.json").open(encoding="utf-8") as f:
        info = json.load(f)
    info["total_episodes"] = 1
    info["total_frames"] = int(ep_meta["length"])
    info["total_videos"] = NUM_CAMERAS
    info["splits"] = {"train": "0:1"}
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    return {
        "episode_index": episode_index,
        "length": int(ep_meta["length"]),
        "tasks": ep_meta.get("tasks"),
        "path": str(out_root),
    }
