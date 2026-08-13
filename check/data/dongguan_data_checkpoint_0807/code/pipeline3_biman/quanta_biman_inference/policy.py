"""Load finetuned biman QLoRA GR00T policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from quanta_biman_inference.constants import BASE_MODEL, EMBODIMENT_TAG_VALUE, NUM_INFERENCE_TIMESTEPS
from quanta_biman_inference.env import ensure_gr00t_imports


def resolve_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir() and path.name.startswith("checkpoint-"):
        return path
    if path.is_dir():
        checkpoints = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        return checkpoints[-1] if checkpoints else None
    return path if path.exists() else None


def load_policy(
    checkpoint_path: Path | str,
    *,
    base_model_path: Path | str | None = None,
    device: str | torch.device | None = None,
    num_inference_timesteps: int = NUM_INFERENCE_TIMESTEPS,
) -> Any:
    ensure_gr00t_imports()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t_qlora_hooks import build_gr00t_policy

    checkpoint_path = Path(checkpoint_path)
    base_model_path = Path(base_model_path or BASE_MODEL)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = build_gr00t_policy(
        checkpoint_path=checkpoint_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device=device,
        base_model_path=base_model_path,
    )
    policy.model.action_head.num_inference_timesteps = num_inference_timesteps

    embodiment_value = policy.embodiment_tag.value
    if embodiment_value != EMBODIMENT_TAG_VALUE:
        raise RuntimeError(
            f"Unexpected embodiment tag {embodiment_value!r}, expected {EMBODIMENT_TAG_VALUE!r}"
        )
    return policy
