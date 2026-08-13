"""Open-loop evaluation on LeRobot data (offline inference smoke)."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from quanta_x1_inference.constants import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EXECUTION_HORIZON,
    GR00T_REPO,
    INFERENCE_TMP,
)
from quanta_x1_inference.dataset_paths import resolve_dataset_path
from quanta_x1_inference.env import ensure_gr00t_imports
from quanta_x1_inference.policy import load_policy, resolve_checkpoint


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_registered_modality_configs():
    """Load quanta_x1_config for LeRobotEpisodeLoader (dataset-side keys)."""
    ensure_gr00t_imports()
    modality_config = GR00T_REPO / "examples/QuantaX1/quanta_x1_config.py"
    sys.path.insert(0, str(modality_config.parent))
    if "quanta_x1_config" in sys.modules:
        importlib.reload(sys.modules["quanta_x1_config"])
    else:
        importlib.import_module("quanta_x1_config")
    import quanta_x1_config

    return quanta_x1_config.quanta_x1_config


def run_open_loop(
    *,
    model_path: Path | str | None,
    dataset_path: Path | str,
    loader_index: int = 0,
    steps: int = 200,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    plot_path: Path | None = None,
) -> dict[str, Any]:
    ensure_gr00t_imports()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import evaluate_single_trajectory

    dataset_path = Path(dataset_path)
    model_path = resolve_checkpoint(Path(model_path) if model_path is not None else DEFAULT_CHECKPOINT)
    if model_path is None:
        raise FileNotFoundError("No checkpoint resolved for open-loop eval")

    modality_configs = get_registered_modality_configs()
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
    )
    if loader_index >= len(loader):
        raise IndexError(f"loader_index {loader_index} out of range for {len(loader)} episodes")

    ep_index = int(loader.episodes_metadata[loader_index]["episode_index"])
    policy = load_policy(model_path)

    if plot_path is None:
        plot_path = INFERENCE_TMP / "plots" / f"open_loop_{model_path.name}.jpeg"
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    mse, mae = evaluate_single_trajectory(
        policy,
        loader,
        loader_index,
        EmbodimentTag.NEW_EMBODIMENT,
        modality_keys=None,
        steps=steps,
        execution_horizon=execution_horizon,
        save_plot_path=str(plot_path),
    )
    return {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "loader_index": loader_index,
        "episode_index": ep_index,
        "device": str(next(policy.model.parameters()).device),
        "steps": steps,
        "execution_horizon": execution_horizon,
        "mse": float(mse),
        "mae": float(mae),
        "plot_path": str(plot_path),
        "ok": bool(np.isfinite(mse) and np.isfinite(mae)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quanta X1 open-loop eval (train-aligned).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Finetuned checkpoint dir (default: checkpoint-5000).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="val",
        help="Dataset alias (val/train/full) or path.",
    )
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument("--tag", type=str, default="ckpt5000_val")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path.",
    )
    args = parser.parse_args()

    output_path = args.output or (INFERENCE_TMP / f"open_loop_{args.tag}.json")
    plot_path = INFERENCE_TMP / "plots" / f"open_loop_{args.tag}.jpeg"

    report = run_open_loop(
        model_path=args.checkpoint,
        dataset_path=resolve_dataset_path(args.dataset_path),
        loader_index=args.loader_index,
        steps=args.steps,
        execution_horizon=args.execution_horizon,
        plot_path=plot_path,
    )
    write_json(output_path, report)

    print(f"Open-loop [{args.tag}]: ok={report['ok']}")
    print(f"  ep_index={report['episode_index']} mse={report['mse']:.6f} mae={report['mae']:.6f}")
    print(f"  plot: {plot_path}")
    print(f"  report: {output_path}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
