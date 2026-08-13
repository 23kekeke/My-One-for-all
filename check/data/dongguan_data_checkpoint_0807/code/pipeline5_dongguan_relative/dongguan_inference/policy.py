"""Load finetuned Dongguan relative QLoRA GR00T policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dongguan_inference.constants import BASE_MODEL, EMBODIMENT_TAG_VALUE, NUM_INFERENCE_TIMESTEPS
from dongguan_inference.env import ensure_dongguan_infer_imports


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


def build_gr00t_policy(
    *,
    checkpoint_path: str | Path,
    embodiment_tag,
    device: str | torch.device,
    base_model_path: str | Path | None = None,
) -> Any:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.policy import BasePolicy
    from gr00t_qlora_hooks import is_qlora_checkpoint, load_qlora_finetuned_model
    from transformers import AutoModel, AutoProcessor

    checkpoint_path = Path(checkpoint_path)
    if isinstance(embodiment_tag, str):
        embodiment_tag = EmbodimentTag.resolve(embodiment_tag)

    processor_dir = (
        checkpoint_path / "processor"
        if (checkpoint_path / "processor").is_dir()
        and not (checkpoint_path / "processor_config.json").exists()
        else checkpoint_path
    )

    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    BasePolicy.__init__(policy, strict=True)
    policy.processor = AutoProcessor.from_pretrained(str(processor_dir))
    policy.processor.eval()
    policy.embodiment_tag = embodiment_tag

    if is_qlora_checkpoint(checkpoint_path):
        if base_model_path is None:
            raise ValueError("base_model_path required for QLoRA checkpoint inference")
        policy.model = load_qlora_finetuned_model(
            checkpoint_path,
            base_model_path=base_model_path,
            device=device,
        )
    else:
        policy.model = AutoModel.from_pretrained(str(checkpoint_path))
        policy.model.eval()
        policy.model.to(device=device, dtype=torch.bfloat16)

    all_modality_configs = policy.processor.get_modality_configs()
    if policy.embodiment_tag.value not in all_modality_configs:
        raise ValueError(
            f"Embodiment tag {policy.embodiment_tag.value!r} missing from checkpoint processor"
        )
    policy.modality_configs = {
        k: v
        for k, v in all_modality_configs[policy.embodiment_tag.value].items()
        if k != "rl_info"
    }
    policy.collate_fn = policy.processor.collator
    language_keys = policy.modality_configs["language"].modality_keys
    language_delta_indices = policy.modality_configs["language"].delta_indices
    assert len(language_keys) >= 1
    assert len(language_delta_indices) == 1
    policy.language_key = language_keys[0]
    return policy


def load_policy(
    checkpoint_path: Path | str,
    *,
    base_model_path: Path | str | None = None,
    device: str | torch.device | None = None,
    num_inference_timesteps: int = NUM_INFERENCE_TIMESTEPS,
) -> Any:
    ensure_dongguan_infer_imports()

    from gr00t.data.embodiment_tags import EmbodimentTag

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
