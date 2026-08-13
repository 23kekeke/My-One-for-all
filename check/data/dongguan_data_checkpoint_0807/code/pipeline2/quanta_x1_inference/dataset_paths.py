"""Dataset path helpers."""

from __future__ import annotations

from pathlib import Path

from quanta_x1_inference.constants import FULL_DATASET, TRAIN_DATASET, VAL_DATASET


def resolve_dataset_path(name_or_path: str | Path) -> Path:
    key = str(name_or_path).strip().lower()
    aliases = {
        "val": VAL_DATASET,
        "validation": VAL_DATASET,
        "train": TRAIN_DATASET,
        "full": FULL_DATASET,
    }
    if key in aliases:
        return aliases[key]
    path = Path(name_or_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {name_or_path}")
    return path
