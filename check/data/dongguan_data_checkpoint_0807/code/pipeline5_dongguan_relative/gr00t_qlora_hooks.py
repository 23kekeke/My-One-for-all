"""4-bit QLoRA hooks for pipeline5: LLM + ViT LoRA (r=32) on frozen NF4 backbone."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

import torch

_QLORA_PATCH_VERSION = "pipeline5_qlora_v1"
_APPLIED = False

DEFAULT_LLM_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_VIT_LORA_TARGET_MODULES = (
    "qkv",
    "proj",
    "linear_fc1",
    "linear_fc2",
)

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05


def qlora_enabled() -> bool:
    return os.environ.get("PIPELINE2_QLORA", "0") == "1"


def bnb_4bit_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _resolve_vlm_model_name(model) -> str:
    local = os.environ.get("GR00T_COSMOS_MODEL_PATH")
    if local and Path(local).is_dir():
        return str(Path(local).resolve())
    return model.config.model_name


def _build_4bit_vlm(model, transformers_loading_kwargs: dict[str, Any] | None):
    from gr00t.model.modules.qwen3_backbone import Qwen3VLForConditionalGeneration

    model_name = _resolve_vlm_model_name(model)
    select_layer = model.config.select_layer
    use_flash_attention = model.config.use_flash_attention

    extra_kwargs: dict[str, Any] = {}
    if use_flash_attention:
        try:
            import flash_attn  # noqa: F401

            extra_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            extra_kwargs["attn_implementation"] = "sdpa"

    load_kwargs = dict(transformers_loading_kwargs or {})
    load_kwargs["quantization_config"] = bnb_4bit_config()

    print(
        "[pipeline5_qlora] loading VLM in 4-bit NF4: "
        f"{model_name!r}",
        flush=True,
    )
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **extra_kwargs,
        **load_kwargs,
    ).eval()

    while len(vlm.language_model.layers) > select_layer:
        vlm.language_model.layers.pop(-1)

    return vlm


def install_qlora_hooks(
    *,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
) -> None:
    """Patch GR00T load path: 4-bit VLM + PEFT LoRA on LLM and ViT linears."""
    global _APPLIED
    if _APPLIED:
        return

    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    orig_create_model = Gr00tN1d7Pipeline._create_model

    def patched_create_model(self):
        model = orig_create_model(self)
        if qlora_enabled():
            apply_qlora_to_gr00t_model(
                model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        return model

    patched_create_model._pipeline5_qlora_patch_version = _QLORA_PATCH_VERSION
    Gr00tN1d7Pipeline._create_model = patched_create_model
    _APPLIED = True


def _action_head_tune_flags(model) -> tuple[bool, bool, bool]:
    cfg = model.config
    return (
        bool(getattr(cfg, "tune_projector", True)),
        bool(getattr(cfg, "tune_diffusion_model", True)),
        bool(getattr(cfg, "tune_vlln", True)),
    )


def _count_lora_params(module, *, name_prefix: str = "") -> int:
    total = 0
    for name, param in module.named_parameters():
        full = f"{name_prefix}.{name}" if name_prefix else name
        if param.requires_grad and "lora" in full.lower():
            total += param.numel()
    return total


def apply_qlora_to_gr00t_model(
    model,
    *,
    transformers_loading_kwargs: dict[str, Any] | None = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    backbone = model.backbone

    old_vlm = backbone.model
    backbone.model = None
    del old_vlm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vlm = _build_4bit_vlm(model, transformers_loading_kwargs)
    if hasattr(vlm, "language_model"):
        vlm.language_model.requires_grad_(False)
    if hasattr(vlm, "visual"):
        vlm.visual.requires_grad_(False)

    vlm = prepare_model_for_kbit_training(vlm)

    combined_targets = list(
        dict.fromkeys(DEFAULT_LLM_LORA_TARGET_MODULES + DEFAULT_VIT_LORA_TARGET_MODULES)
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=combined_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    vlm = get_peft_model(vlm, lora_config)
    backbone.model = vlm

    for name, param in vlm.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    if hasattr(model, "action_head"):
        tune_projector, tune_diffusion_model, tune_vlln = _action_head_tune_flags(model)
        model.action_head.set_trainable_parameters(
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora_trainable = sum(
        p.numel() for n, p in vlm.named_parameters() if p.requires_grad and "lora" in n.lower()
    )
    llm_lora_trainable = _count_lora_params(vlm.language_model, name_prefix="language_model")
    vit_lora_trainable = _count_lora_params(vlm.visual, name_prefix="visual") if hasattr(vlm, "visual") else 0

    groups = summarize_trainable_groups(model)
    report = {
        "qlora_applied": True,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "llm_lora_target_modules": list(DEFAULT_LLM_LORA_TARGET_MODULES),
        "vit_lora_target_modules": list(DEFAULT_VIT_LORA_TARGET_MODULES),
        "trainable_params": trainable,
        "total_params": total,
        "lora_trainable_params": lora_trainable,
        "llm_lora_trainable_params": llm_lora_trainable,
        "vit_lora_trainable_params": vit_lora_trainable,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        "trainable_groups": groups,
    }
    print(
        "[pipeline5_qlora] applied PEFT LoRA on LLM+ViT; "
        f"trainable={trainable:,} ({report['trainable_pct']:.2f}%), "
        f"lora_total={lora_trainable:,}, llm_lora={llm_lora_trainable:,}, vit_lora={vit_lora_trainable:,}, "
        f"groups={groups}",
        flush=True,
    )
    if groups["backbone_lora"] == 0:
        raise RuntimeError("No trainable LoRA adapter parameters found on VLM backbone")
    if vit_lora_trainable == 0:
        raise RuntimeError("No trainable ViT LoRA parameters found (expected visual qkv/proj/fc LoRA)")
    return report


def summarize_trainable_groups(model) -> dict[str, int]:
    groups = {
        "backbone_lora": 0,
        "backbone_llm_lora": 0,
        "backbone_vit_lora": 0,
        "backbone_other": 0,
        "action_head_projector": 0,
        "action_head_dit": 0,
        "action_head_other": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if "backbone" in name:
            if "lora" in name.lower():
                groups["backbone_lora"] += n
                if ".visual." in name:
                    groups["backbone_vit_lora"] += n
                elif ".language_model." in name:
                    groups["backbone_llm_lora"] += n
            else:
                groups["backbone_other"] += n
        elif "action_head" in name:
            if ".model." in name and "action_head" in name:
                groups["action_head_dit"] += n
            elif any(
                key in name
                for key in (
                    "state_encoder",
                    "action_encoder",
                    "action_decoder",
                    "position_embedding",
                    "vlln",
                    "vl_self_attention",
                )
            ):
                groups["action_head_projector"] += n
            else:
                groups["action_head_other"] += n
        else:
            groups["other"] += n
    return groups


def is_qlora_checkpoint(model_path: str | Path) -> bool:
    model_dir = Path(model_path)
    if (model_dir / "adapter_config.json").exists():
        return True
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        import json

        data = json.loads(index_path.read_text(encoding="utf-8"))
        return any("lora_" in key for key in data.get("weight_map", {}))
    return False


def _load_sharded_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    import json

    from safetensors.torch import load_file

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(model_dir / shard))
        return state_dict

    single = model_dir / "model.safetensors"
    if single.is_file():
        return load_file(single)

    raise FileNotFoundError(f"No model weights found under {model_dir}")


def load_qlora_finetuned_model(
    checkpoint_dir: str | Path,
    *,
    base_model_path: str | Path,
    device: str | torch.device = "cuda",
):
    """Rebuild 4-bit+LoRA backbone, then load finetuned checkpoint weights."""
    import gr00t.model  # noqa: F401
    from transformers import AutoModel

    checkpoint_dir = Path(checkpoint_dir)
    base_model_path = Path(base_model_path)

    install_qlora_hooks()
    os.environ["PIPELINE2_QLORA"] = "1"

    print(
        f"[pipeline5_qlora] rebuilding QLoRA model from base={base_model_path!r} "
        f"checkpoint={checkpoint_dir!r}",
        flush=True,
    )
    model = AutoModel.from_pretrained(str(base_model_path))
    apply_qlora_to_gr00t_model(model)

    state_dict = _load_sharded_state_dict(checkpoint_dir)
    load_result = model.load_state_dict(state_dict, strict=False)
    print(
        "[pipeline5_qlora] loaded finetuned checkpoint: "
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}",
        flush=True,
    )

    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return model
