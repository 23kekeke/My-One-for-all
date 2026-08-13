#!/usr/bin/env python3
"""
Pipeline7 dongguan eef-only relative-action finetune entry.

- State 32D (eef + gripper + joint ×2); action 20D (eef + gripper only, RELATIVE eef).
- Parquet stays absolute (action[t]=state[t+1]); relative conversion at train time.
- LLM + ViT QLoRA r=32 on 4-bit NF4 VLM (tune_llm/tune_visual remain False).
- delta_indices [-7,-1,0], state_history_length=3.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GR00T_REPO = ROOT / "isaacGr00t"
PIPELINE2 = ROOT / "pipeline2"
PIPELINE3 = ROOT / "pipeline3_biman"
PIPELINE7 = Path(__file__).resolve().parent

sys.path[:0] = [str(PIPELINE7), str(GR00T_REPO), str(PIPELINE2), str(PIPELINE3)]

os.environ.setdefault("PIPELINE2_QLORA", "1")

import dongguan_eef_only_relative_config

from gr00t_dongguan_hooks import install_dongguan_state_history_hooks
from gr00t_offline_hooks import install_offline_hooks
from gr00t_qlora_hooks import DEFAULT_LORA_ALPHA, DEFAULT_LORA_R, install_qlora_hooks

install_offline_hooks()
install_dongguan_state_history_hooks(
    state_history_length=dongguan_eef_only_relative_config.STATE_HISTORY_LENGTH,
)
if os.environ.get("PIPELINE2_QLORA", "0") == "1":
    install_qlora_hooks(lora_r=DEFAULT_LORA_R, lora_alpha=DEFAULT_LORA_ALPHA)

import gr00t.experiment.experiment as experiment_module

_original_run = experiment_module.run


def _run_with_dongguan_eef_only_relative_config(config):
    config.model.letter_box_transform = True
    config.model.shortest_image_edge = 256
    config.model.crop_fraction = 1.0
    config.model.image_crop_size = None
    config.model.image_target_size = None

    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_top_llm_layers = 0

    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_vlln = True

    if os.environ.get("PIPELINE2_QLORA", "0") == "1":
        config.model.load_bf16 = False
    if config.model.tune_diffusion_model:
        config.training.optim = "paged_adamw_8bit"

    config.training.global_batch_size = max(1, int(config.training.global_batch_size))

    config.data.allow_padding = True
    config.model.state_history_length = dongguan_eef_only_relative_config.STATE_HISTORY_LENGTH
    config.model.use_relative_action = True

    from train_logging_utils import (
        build_run_manifest,
        resolve_output_dir,
        write_run_manifest,
        write_training_summary,
    )

    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output_dir, build_run_manifest(config))

    print(
        "[pipeline7_dongguan_eef_only_finetune] "
        f"qlora={os.environ.get('PIPELINE2_QLORA', '0')} "
        f"lora_r={DEFAULT_LORA_R} vit_lora=True "
        f"use_relative_action={config.model.use_relative_action} "
        f"action_reps={dongguan_eef_only_relative_config.ACTION_REPS} "
        f"letter_box_transform={config.model.letter_box_transform} "
        f"shortest_image_edge={config.model.shortest_image_edge} "
        f"observation_delta_indices={dongguan_eef_only_relative_config.OBSERVATION_DELTA_INDICES} "
        f"state_history_length={config.model.state_history_length} "
        f"allow_padding={config.data.allow_padding} "
        f"tune_projector={config.model.tune_projector} "
        f"tune_diffusion_model={config.model.tune_diffusion_model} "
        f"tune_llm={config.model.tune_llm} tune_visual={config.model.tune_visual} "
        f"optim={config.training.optim} "
        f"global_batch_size={config.training.global_batch_size} "
        f"gradient_accumulation_steps={config.training.gradient_accumulation_steps}",
        flush=True,
    )
    try:
        return _original_run(config)
    except Exception:
        write_training_summary(output_dir, status="failed")
        raise
    else:
        write_training_summary(output_dir, status="completed")


experiment_module.run = _run_with_dongguan_eef_only_relative_config

runpy.run_path(str(GR00T_REPO / "gr00t/experiment/launch_finetune.py"), run_name="__main__")
