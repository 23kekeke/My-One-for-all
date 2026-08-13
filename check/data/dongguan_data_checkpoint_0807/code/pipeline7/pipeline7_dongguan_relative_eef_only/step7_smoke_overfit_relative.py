#!/usr/bin/env python3
"""Step 7: 1-episode QLoRA overfit smoke (LLM+ViT r=32, eef-only relative action)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_utils import SMOKE_ROOT, write_json
from train_logging_utils import extract_loss_series, summarize_loss
from train_smoke_utils import (
    BASE_MODEL,
    EP0_DATASET,
    GR00T_PYTHON,
    MULTI_DATASET,
    OVERFIT_OUTPUT,
    RELATIVE_FINETUNE_ENTRY,
    RELATIVE_MODALITY_CONFIG,
    create_single_episode_view,
    setup_gr00t_env,
)

# ViT LoRA + 3-frame history uses more VRAM; probe smaller batch first.
BATCH_PROBE_ORDER = ((2, 2), (4, 2))
LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")


def build_overfit_command(
    *,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int,
    save_steps: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    skip_weight_loading: bool,
) -> list[str]:
    return [
        str(GR00T_PYTHON),
        str(RELATIVE_FINETUNE_ENTRY),
        "--base-model-path",
        str(BASE_MODEL),
        "--dataset-path",
        str(dataset_path),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(RELATIVE_MODALITY_CONFIG),
        "--output-dir",
        str(output_dir),
        "--num-gpus",
        "1",
        "--max-steps",
        str(max_steps),
        "--save-steps",
        str(save_steps),
        "--save-total-limit",
        "2",
        "--global-batch-size",
        str(global_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--dataloader-num-workers",
        "0",
        "--episode-sampling-rate",
        "1.0",
        "--num-shards-per-epoch",
        "16",
        "--shard-size",
        "128",
        "--learning-rate",
        "1e-4",
        "--no-tune-llm",
        "--no-tune-visual",
        "--tune-projector",
        "--tune-diffusion-model",
        "--state-dropout-prob",
        "0.0",
        *(
            ["--skip-weight-loading"]
            if skip_weight_loading
            else []
        ),
    ]


def verify_overfit_logs(stdout: str, stderr: str) -> dict:
    combined = f"{stdout}\n{stderr}"
    return {
        "qlora_4bit_logged": "[pipeline7_qlora] loading VLM in 4-bit NF4" in combined,
        "vit_lora_logged": "vit_lora=True" in combined,
        "llm_vit_lora_applied": "applied PEFT LoRA on LLM+ViT" in combined,
        "relative_finetune_logged": "[pipeline7_dongguan_eef_only_finetune]" in combined,
        "use_relative_action_logged": "use_relative_action=True" in combined,
        "action_reps_logged": "RELATIVE" in combined and "ABSOLUTE" in combined,
        "delta_indices_logged": "[-7, -1, 0]" in combined,
        "completed": "Training completed" in combined or re.search(r"train_loss", combined) is not None,
    }


def parse_losses_from_text(text: str) -> list[float]:
    return [float(x) for x in LOSS_RE.findall(text)]


def summarize_overfit_loss(output_dir: Path, stdout: str, stderr: str) -> dict:
    series = extract_loss_series(output_dir)
    summary = summarize_loss(series)
    if summary.get("count", 0) == 0:
        losses = parse_losses_from_text(f"{stdout}\n{stderr}")
        if losses:
            summary = {
                "count": len(losses),
                "first": losses[0],
                "last": losses[-1],
                "min": min(losses),
                "max": max(losses),
                "source": "stdout_regex",
            }
    else:
        summary["source"] = "trainer_state"
    if summary.get("count", 0) >= 2:
        first = float(summary["first"])
        last = float(summary["last"])
        summary["loss_drop"] = first - last
        summary["loss_drop_ratio"] = (first - last) / max(first, 1e-6)
        summary["loss_decreased"] = last < first
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 eef-only relative 1-ep overfit smoke (Step 7).")
    parser.add_argument("--source-dataset-path", type=Path, default=MULTI_DATASET)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--save-steps", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=OVERFIT_OUTPUT)
    parser.add_argument("--skip-weight-loading", action="store_true")
    args = parser.parse_args()

    ep_view = create_single_episode_view(
        args.episode_index,
        source_root=args.source_dataset_path,
        out_root=EP0_DATASET,
    )

    env = setup_gr00t_env()
    last_result = None
    attempts: list[dict] = []

    for gbs, gas in BATCH_PROBE_ORDER:
        cmd = build_overfit_command(
            dataset_path=Path(ep_view["path"]),
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            global_batch_size=gbs,
            gradient_accumulation_steps=gas,
            skip_weight_loading=args.skip_weight_loading,
        )
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log_checks = verify_overfit_logs(proc.stdout, proc.stderr)
        loss_summary = summarize_overfit_loss(args.output_dir, proc.stdout, proc.stderr)
        attempt = {
            "returncode": proc.returncode,
            "global_batch_size": gbs,
            "gradient_accumulation_steps": gas,
            "log_checks": log_checks,
            "loss_summary": loss_summary,
            "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
            "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        }
        attempts.append(attempt)
        last_result = attempt
        if proc.returncode == 0:
            break

    loss_ok = bool(
        last_result
        and last_result.get("loss_summary", {}).get("loss_decreased")
    )
    report = {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "step": "step7_smoke_overfit_relative",
        "episode_view": ep_view,
        "output_dir": str(args.output_dir),
        "attempts": attempts,
        "last_run": last_result,
        "loss_ok": loss_ok,
        "ok": bool(last_result and last_result["returncode"] == 0 and loss_ok),
    }
    write_json(SMOKE_ROOT / "overfit_report.json", report)

    print(f"Step7 relative overfit smoke: ok={report['ok']}")
    if last_result:
        print(f"  batch: gbs={last_result['global_batch_size']} gas={last_result['gradient_accumulation_steps']}")
        print(f"  log_checks: {last_result['log_checks']}")
        print(f"  loss_summary: {last_result.get('loss_summary')}")
    print(f"  report: {SMOKE_ROOT / 'overfit_report.json'}")
    if not report["ok"]:
        if last_result and last_result.get("stderr_tail"):
            print("\n--- stderr tail ---")
            print(last_result["stderr_tail"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
