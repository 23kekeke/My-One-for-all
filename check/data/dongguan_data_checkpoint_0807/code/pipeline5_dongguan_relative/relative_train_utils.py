"""Shared helpers for pipeline5_dongguan_relative GR00T training."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GR00T_REPO = ROOT / "isaacGr00t"
PIPELINE2 = ROOT / "pipeline2"
PIPELINE3 = ROOT / "pipeline3_biman"
PIPELINE5 = Path(__file__).resolve().parent

BASE_MODEL = ROOT / "GR00T-N1.7-3B"
COSMOS_MODEL = ROOT / "Cosmos-Reason2-2B"
CHECKPOINT_ROOT = Path("/data/dongguan_data_checkpoint_0807")
MULTI_DATASET = CHECKPOINT_ROOT / "lerobot/multi_345"
SMOKE_ROOT = CHECKPOINT_ROOT / "tmp/step_smoke"

GR00T_PYTHON = GR00T_REPO / ".venv/bin/python"
RELATIVE_MODALITY_CONFIG = PIPELINE5 / "dongguan_relative_modality.py"

EMBODIMENT_TAG_VALUE = "new_embodiment"
ACTION_HORIZON = 40
NUM_CAMERAS = 3

EXPECTED_ACTION_REPS = [
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
    "RELATIVE",
    "ABSOLUTE",
    "RELATIVE",
]
RELATIVE_ACTION_KEYS = (
    "left_eef_9d",
    "left_joint_position",
    "right_eef_9d",
    "right_joint_position",
)


def setup_gr00t_env() -> dict[str, str]:
    env = os.environ.copy()
    pipeline_paths = [str(GR00T_REPO), str(PIPELINE2), str(PIPELINE3), str(PIPELINE5)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(pipeline_paths + ([existing] if existing else []))
    env["GR00T_COSMOS_MODEL_PATH"] = str(COSMOS_MODEL)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["LOGURU_LEVEL"] = env.get("LOGURU_LEVEL", "INFO")
    env["GROOT_PATCH_MISTRAL"] = "1"
    env["GROOT_HF_LOCAL_FIRST"] = "1"
    env["PIPELINE2_QLORA"] = "1"
    return env


def ensure_gr00t_imports() -> None:
    env = setup_gr00t_env()
    for key, value in env.items():
        os.environ[key] = value

    for path in (str(PIPELINE5), str(PIPELINE3), str(PIPELINE2), str(GR00T_REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)

    for mod_name in list(sys.modules):
        if mod_name in {
            "dongguan_relative_config",
            "quanta_biman_relative_config",
            "quanta_biman_config",
            "gr00t",
            "gr00t_qlora_hooks",
        } or mod_name.startswith("gr00t."):
            del sys.modules[mod_name]

    sys.path.insert(0, str(PIPELINE5))
    from gr00t_offline_hooks import install_offline_hooks
    from gr00t_qlora_hooks import DEFAULT_LORA_ALPHA, DEFAULT_LORA_R, install_qlora_hooks

    install_offline_hooks()
    if os.environ.get("PIPELINE2_QLORA", "0") == "1":
        install_qlora_hooks(lora_r=DEFAULT_LORA_R, lora_alpha=DEFAULT_LORA_ALPHA)


def load_relative_modality_config() -> None:
    ensure_gr00t_imports()
    if "dongguan_relative_config" in sys.modules:
        importlib.reload(sys.modules["dongguan_relative_config"])
    else:
        importlib.import_module("dongguan_relative_config")


def get_registered_modality_configs():
    load_relative_modality_config()
    import dongguan_relative_config

    return dongguan_relative_config.dongguan_relative_config


def load_tasks_jsonl(dataset_path: Path) -> dict[int, str]:
    tasks_path = dataset_path / "meta/tasks.jsonl"
    mapping: dict[int, str] = {}
    with tasks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            mapping[int(entry["task_index"])] = str(entry["task"])
    return mapping


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_dataset_meta(dataset_path: Path) -> dict[str, Any]:
    with (dataset_path / "meta/info.json").open(encoding="utf-8") as f:
        return json.load(f)


def check_relative_stats(dataset_path: Path) -> dict[str, Any]:
    rel_path = dataset_path / "meta/relative_stats.json"
    if not rel_path.is_file():
        return {"ok": False, "reason": "missing relative_stats.json", "path": str(rel_path)}

    rel_stats = json.loads(rel_path.read_text(encoding="utf-8"))
    missing = [k for k in RELATIVE_ACTION_KEYS if k not in rel_stats]
    return {
        "ok": not missing,
        "path": str(rel_path),
        "keys_present": sorted(k for k in rel_stats if not k.startswith("__")),
        "missing_keys": missing,
    }
