"""Training run logging and cross-run comparison for pipeline7."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dongguan_eef_only_relative_config
from gr00t_qlora_hooks import DEFAULT_LORA_ALPHA, DEFAULT_LORA_R
from manifest_utils import OUTPUT_ROOT, write_json

RUNS_INDEX = OUTPUT_ROOT / "runs_index.jsonl"


def resolve_output_dir(config) -> Path:
    if config.training.experiment_name is None:
        return Path(config.training.output_dir)
    return Path(config.training.output_dir) / config.training.experiment_name


def _git_commit_short() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except OSError:
        pass
    return None


def _dataset_summary(dataset_path: str) -> dict[str, Any]:
    root = Path(dataset_path.split(os.pathsep)[0])
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        return {"dataset_path": dataset_path, "info_found": False}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return {
        "dataset_path": str(root.resolve()),
        "info_found": True,
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "fps": info.get("fps"),
    }


def build_run_manifest(config) -> dict[str, Any]:
    output_dir = resolve_output_dir(config)
    dataset_path = config.data.datasets[0].dataset_paths[0] if config.data.datasets else ""
    return {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "run_id": output_dir.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": sys.argv,
        "git_commit": _git_commit_short(),
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": _dataset_summary(dataset_path),
        "training_contract": {
            "state_dim": 32,
            "action_dim": 20,
            "action_keys": [
                "left_eef_9d",
                "left_gripper_position",
                "right_eef_9d",
                "right_gripper_position",
            ],
            "action_parquet_semantics": "absolute action[t]=state[t+1]",
            "train_action_mode": "RELATIVE (eef only) + ABSOLUTE (gripper); no joint in action",
            "use_relative_action": True,
            "observation_delta_indices": dongguan_eef_only_relative_config.OBSERVATION_DELTA_INDICES,
            "state_history_length": dongguan_eef_only_relative_config.STATE_HISTORY_LENGTH,
            "action_reps": dongguan_eef_only_relative_config.ACTION_REPS,
            "allow_padding": bool(getattr(config.data, "allow_padding", True)),
        },
        "qlora": {
            "enabled": os.environ.get("PIPELINE2_QLORA", "0") == "1",
            "lora_r": DEFAULT_LORA_R,
            "lora_alpha": DEFAULT_LORA_ALPHA,
            "vit_lora": True,
            "tune_llm_native": bool(config.model.tune_llm),
            "tune_visual_native": bool(config.model.tune_visual),
        },
        "hyperparameters": {
            "base_model_path": config.training.start_from_checkpoint,
            "output_dir": str(output_dir.resolve()),
            "max_steps": config.training.max_steps,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "global_batch_size": config.training.global_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "accumulated_batch_size": config.training.accumulated_batch_size,
            "learning_rate": config.training.learning_rate,
            "warmup_ratio": config.training.warmup_ratio,
            "weight_decay": config.training.weight_decay,
            "optim": config.training.optim,
            "logging_steps": config.training.logging_steps,
            "state_dropout_prob": config.model.state_dropout_prob,
            "episode_sampling_rate": config.data.episode_sampling_rate,
            "num_shards_per_epoch": config.data.num_shards_per_epoch,
            "shard_size": config.data.shard_size,
            "num_gpus": config.training.num_gpus,
            "use_wandb": config.training.use_wandb,
            "wandb_project": config.training.wandb_project,
            "experiment_name": config.training.experiment_name,
        },
        "model_tuning": {
            "tune_projector": config.model.tune_projector,
            "tune_diffusion_model": config.model.tune_diffusion_model,
            "tune_vlln": config.model.tune_vlln,
            "letter_box_transform": config.model.letter_box_transform,
            "shortest_image_edge": config.model.shortest_image_edge,
            "crop_fraction": config.model.crop_fraction,
        },
    }


def write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / "run_manifest.json"
    write_json(path, manifest)
    return path


def _find_trainer_state_paths(output_dir: Path) -> list[Path]:
    paths = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    root = output_dir / "trainer_state.json"
    if root.is_file():
        paths.append(root)
    return paths


def extract_loss_series(output_dir: Path) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for state_path in _find_trainer_state_paths(output_dir):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        step = None
        name = state_path.parent.name
        if name.startswith("checkpoint-"):
            step = int(name.split("-", 1)[1])
        for entry in state.get("log_history", []):
            if "loss" in entry:
                series.append(
                    {
                        "checkpoint_step": step,
                        "global_step": entry.get("step"),
                        "epoch": entry.get("epoch"),
                        "loss": entry.get("loss"),
                        "learning_rate": entry.get("learning_rate"),
                        "grad_norm": entry.get("grad_norm"),
                    }
                )
    by_step: dict[int, dict[str, Any]] = {}
    for row in series:
        gs = row.get("global_step")
        if gs is not None:
            by_step[int(gs)] = row
    return [by_step[k] for k in sorted(by_step)]


def summarize_loss(loss_series: list[dict[str, Any]]) -> dict[str, Any]:
    if not loss_series:
        return {"count": 0}
    losses = [float(x["loss"]) for x in loss_series if x.get("loss") is not None]
    if not losses:
        return {"count": 0}
    n = len(losses)
    quarter = max(1, n // 4)
    return {
        "count": n,
        "first": losses[0],
        "last": losses[-1],
        "min": min(losses),
        "max": max(losses),
        "mean_first_quarter": sum(losses[:quarter]) / quarter,
        "mean_last_quarter": sum(losses[-quarter:]) / quarter,
        "final_global_step": loss_series[-1].get("global_step"),
    }


def write_training_summary(output_dir: Path, *, status: str = "completed") -> Path:
    manifest_path = output_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    loss_series = extract_loss_series(output_dir)
    checkpoints = sorted(
        int(p.parent.name.split("-", 1)[1])
        for p in output_dir.glob("checkpoint-*/trainer_state.json")
    )
    summary = {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "run_id": output_dir.name,
        "output_dir": str(output_dir.resolve()),
        "status": status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints_saved": checkpoints,
        "loss": summarize_loss(loss_series),
        "loss_series_path": str((output_dir / "loss_series.json").resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
        "train_log_path": str((output_dir / "train.log").resolve()),
        "experiment_cfg": str((output_dir / "experiment_cfg").resolve()),
    }
    if manifest:
        summary["hyperparameters"] = manifest.get("hyperparameters")
        summary["training_contract"] = manifest.get("training_contract")

    write_json(output_dir / "loss_series.json", {"series": loss_series})
    write_json(output_dir / "training_summary.json", summary)
    append_runs_index(summary)
    return output_dir / "training_summary.json"


def append_runs_index(summary: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": summary.get("run_id"),
        "output_dir": summary.get("output_dir"),
        "status": summary.get("status"),
        "finished_at_utc": summary.get("finished_at_utc"),
        "loss": summary.get("loss"),
        "checkpoints_saved": summary.get("checkpoints_saved"),
        "hyperparameters": summary.get("hyperparameters"),
    }
    with RUNS_INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare_runs(run_dirs: list[Path]) -> dict[str, Any]:
    rows = []
    for run_dir in run_dirs:
        summary_path = run_dir / "training_summary.json"
        if summary_path.is_file():
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        manifest: dict[str, Any] = {}
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        loss_series = extract_loss_series(run_dir)
        rows.append(
            {
                "run_id": run_dir.name,
                "output_dir": str(run_dir.resolve()),
                "hyperparameters": manifest.get("hyperparameters"),
                "training_contract": manifest.get("training_contract"),
                "loss": summarize_loss(loss_series),
                "checkpoints_saved": sorted(
                    int(p.parent.name.split("-", 1)[1])
                    for p in run_dir.glob("checkpoint-*/trainer_state.json")
                ),
            }
        )
    rows.sort(key=lambda r: r.get("run_id", ""))
    return {"runs_compared": len(rows), "runs": rows}
