"""Pipeline7 QLoRA infer load: ViT+LLM LoRA and state_history_length from checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from gr00t_dongguan_hooks import checkpoint_state_history_length


def _resize_action_head_state_history(model, target_length: int) -> None:
    current_length = int(getattr(model.config, "state_history_length", 1))
    if current_length == target_length:
        return

    action_head = model.action_head
    config = model.config
    old_input_dim = config.max_state_dim * current_length
    new_input_dim = config.max_state_dim * target_length

    print(
        "[pipeline7_dongguan_infer] resizing state_history_length "
        f"{current_length} -> {target_length} "
        f"(state_encoder input_dim {old_input_dim} -> {new_input_dim})",
        flush=True,
    )

    config.state_history_length = target_length
    action_head.config.state_history_length = target_length

    from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP

    new_encoder = CategorySpecificMLP(
        num_categories=config.max_num_embodiments,
        input_dim=new_input_dim,
        hidden_dim=action_head.hidden_size,
        output_dim=action_head.input_embedding_dim,
    )
    old_encoder = action_head.state_encoder
    if current_length < target_length:
        with torch.no_grad():
            new_encoder.layer1.W[:, :old_input_dim].copy_(old_encoder.layer1.W)
            new_encoder.layer1.b.copy_(old_encoder.layer1.b)
            new_encoder.layer2.W.copy_(old_encoder.layer2.W)
            new_encoder.layer2.b.copy_(old_encoder.layer2.b)

    device = next(old_encoder.parameters()).device
    dtype = next(old_encoder.parameters()).dtype
    action_head.state_encoder = new_encoder.to(device=device, dtype=dtype)

    action_head.set_trainable_parameters(
        action_head.tune_projector,
        action_head.tune_diffusion_model,
        action_head.tune_vlln,
    )


def install_dongguan_qlora_infer_hooks() -> None:
    """Patch load_qlora_finetuned_model for checkpoint state_history_length."""
    import gr00t_qlora_hooks as qlora_hooks

    if getattr(qlora_hooks.load_qlora_finetuned_model, "_pipeline7_dongguan_infer_patch", False):
        return

    def patched_load_with_resize(
        checkpoint_dir: str | Path,
        *,
        base_model_path: str | Path,
        device: str | torch.device = "cuda",
    ):
        checkpoint_dir = Path(checkpoint_dir)
        target_length = checkpoint_state_history_length(checkpoint_dir)

        print(
            "[pipeline7_dongguan_infer] QLoRA infer load with state_history_length="
            f"{target_length} checkpoint={checkpoint_dir!r}",
            flush=True,
        )

        import gr00t.model  # noqa: F401
        from transformers import AutoModel

        qlora_hooks.install_qlora_hooks()
        os.environ["PIPELINE2_QLORA"] = "1"

        model = AutoModel.from_pretrained(str(base_model_path))
        _resize_action_head_state_history(model, target_length)
        qlora_hooks.apply_qlora_to_gr00t_model(model)

        state_dict = qlora_hooks._load_sharded_state_dict(checkpoint_dir)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(
            "[pipeline7_qlora] loaded finetuned checkpoint: "
            f"missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}",
            flush=True,
        )

        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        return model

    patched_load_with_resize._pipeline7_dongguan_infer_patch = True
    qlora_hooks.load_qlora_finetuned_model = patched_load_with_resize
