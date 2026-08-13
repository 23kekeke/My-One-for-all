"""Pipeline5 hooks: sync action-head state_history_length with delta_indices [-7,-1,0]."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

_APPLIED = False


def checkpoint_state_history_length(checkpoint_dir: Path | str) -> int:
    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        return 1
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return int(cfg.get("state_history_length", 1))


def _resize_action_head_state_history(model, target_length: int) -> None:
    current_length = int(getattr(model.config, "state_history_length", 1))
    if current_length == target_length:
        return

    action_head = model.action_head
    config = model.config
    old_input_dim = config.max_state_dim * current_length
    new_input_dim = config.max_state_dim * target_length

    logging.info(
        "[pipeline5_dongguan] resizing state_history_length %s -> %s "
        "(state_encoder input_dim %s -> %s)",
        current_length,
        target_length,
        old_input_dim,
        new_input_dim,
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


def install_dongguan_state_history_hooks(*, state_history_length: int) -> None:
    global _APPLIED
    if _APPLIED:
        return

    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from gr00t.utils.dist_utils import run_or_wait_on_rank0

    orig_create_model = Gr00tN1d7Pipeline._create_model

    def patched_create_model(self):
        model = orig_create_model(self)

        desired = int(getattr(self.config.model, "state_history_length", state_history_length))
        _resize_action_head_state_history(model, desired)

        save_cfg_dir = getattr(self, "save_cfg_dir", None)
        if save_cfg_dir is not None:
            with run_or_wait_on_rank0(label="final_model_config.json rewrite") as is_rank0:
                if is_rank0:
                    path = Path(save_cfg_dir) / "final_model_config.json"
                    path.write_text(model.config.to_filtered_json())

        return model

    patched_create_model._pipeline5_dongguan_state_history_patch = True
    Gr00tN1d7Pipeline._create_model = patched_create_model
    _APPLIED = True
