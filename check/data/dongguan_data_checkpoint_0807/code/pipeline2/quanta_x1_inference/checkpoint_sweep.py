"""Multi-checkpoint open-loop sweep and summary."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

from quanta_x1_inference.constants import (
    DEFAULT_EXECUTION_HORIZON,
    DEFAULT_SWEEP_CHECKPOINT_STEPS,
    DEFAULT_TRAIN_OUTPUT,
    INFERENCE_TMP,
    VAL_DATASET,
)
from quanta_x1_inference.dataset_paths import resolve_dataset_path
from quanta_x1_inference.open_loop import run_open_loop, write_json

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def discover_checkpoints(output_dir: Path | str) -> list[Path]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Training output dir not found: {output_dir}")
    checkpoints = []
    for path in sorted(output_dir.iterdir()):
        if path.is_dir() and _CHECKPOINT_RE.match(path.name):
            checkpoints.append(path)
    return checkpoints


def checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a checkpoint dir: {path}")
    return int(match.group(1))


def select_checkpoints(
    output_dir: Path | str,
    steps: Sequence[int] | None = None,
) -> list[Path]:
    all_ckpts = discover_checkpoints(output_dir)
    if not steps:
        return all_ckpts
    wanted = {int(s) for s in steps}
    selected = [p for p in all_ckpts if checkpoint_step(p) in wanted]
    missing = sorted(wanted - {checkpoint_step(p) for p in selected})
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for steps: {missing} under {output_dir}")
    return sorted(selected, key=checkpoint_step)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def run_checkpoint_sweep(
    *,
    output_dir: Path | str = DEFAULT_TRAIN_OUTPUT,
    checkpoint_steps: Sequence[int] | None = DEFAULT_SWEEP_CHECKPOINT_STEPS,
    dataset_path: Path | str = VAL_DATASET,
    loader_indices: Sequence[int] = (0,),
    steps: int = 200,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    save_plots: bool = False,
    plots_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    dataset_path = resolve_dataset_path(dataset_path)
    checkpoints = select_checkpoints(output_dir, checkpoint_steps)
    plots_dir = plots_dir or (INFERENCE_TMP / "plots" / "sweep")
    reports_dir = reports_dir or (INFERENCE_TMP / "sweep")
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        step = checkpoint_step(ckpt)
        for loader_index in loader_indices:
            tag = f"ckpt{step}_loader{loader_index}"
            plot_path = plots_dir / f"open_loop_{tag}.jpeg" if save_plots else None
            try:
                report = run_open_loop(
                    model_path=ckpt,
                    dataset_path=dataset_path,
                    loader_index=loader_index,
                    steps=steps,
                    execution_horizon=execution_horizon,
                    plot_path=plot_path,
                )
            except Exception as exc:
                report = {
                    "model_path": str(ckpt),
                    "loader_index": loader_index,
                    "ok": False,
                    "error": repr(exc),
                }
            report["checkpoint_step"] = step
            report["tag"] = tag
            write_json(reports_dir / f"{tag}.json", report)
            runs.append(report)

    ok_runs = [r for r in runs if r.get("ok")]
    best_by_mae: dict[str, Any] | None = None
    if ok_runs:
        best_by_mae = min(ok_runs, key=lambda r: float(r["mae"]))

    summary_rows = []
    for report in runs:
        summary_rows.append(
            {
                "checkpoint_step": report.get("checkpoint_step"),
                "loader_index": report.get("loader_index"),
                "episode_index": report.get("episode_index"),
                "mse": report.get("mse"),
                "mae": report.get("mae"),
                "ok": report.get("ok"),
                "error": report.get("error"),
            }
        )

    summary = {
        "output_dir": str(output_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "loader_indices": list(loader_indices),
        "steps": steps,
        "execution_horizon": execution_horizon,
        "checkpoint_steps": [checkpoint_step(p) for p in checkpoints],
        "runs": runs,
        "summary_table": summary_rows,
        "best_by_mae": best_by_mae,
        "ok": bool(ok_runs) and all(r.get("ok") for r in runs),
    }
    return summary


def print_summary_table(summary: dict[str, Any]) -> None:
    print("\ncheckpoint | loader | ep_index | MSE      | MAE      | ok")
    print("-----------+--------+----------+----------+----------+----")
    for row in summary.get("summary_table", []):
        if not row.get("ok"):
            print(
                f"{row.get('checkpoint_step', '?'):>10} | "
                f"{row.get('loader_index', '?'):>6} | "
                f"{'-':>8} | {'-':>8} | {'-':>8} | False"
            )
            continue
        print(
            f"{row['checkpoint_step']:>10} | "
            f"{row['loader_index']:>6} | "
            f"{row['episode_index']:>8} | "
            f"{row['mse']:>8.6f} | "
            f"{row['mae']:>8.6f} | True"
        )
    best = summary.get("best_by_mae")
    if best:
        print(
            f"\nBest by MAE: checkpoint-{best['checkpoint_step']} "
            f"(loader={best['loader_index']}, ep={best['episode_index']}, "
            f"MSE={best['mse']:.6f}, MAE={best['mae']:.6f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep open-loop eval across checkpoints.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help="Training output dir containing checkpoint-* folders.",
    )
    parser.add_argument(
        "--steps-list",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SWEEP_CHECKPOINT_STEPS),
        help="Comma-separated global steps, e.g. 5000,8000,10000,15000.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="val",
        help="Dataset alias (val/train/full) or path.",
    )
    parser.add_argument(
        "--loader-indices",
        type=str,
        default="0",
        help="Comma-separated loader indices into the dataset.",
    )
    parser.add_argument("--steps", type=int, default=200, help="Open-loop timesteps per episode.")
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument("--save-plots", action="store_true", help="Save per-run plots (slower).")
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_TMP / "checkpoint_sweep_summary.json",
    )
    args = parser.parse_args()

    summary = run_checkpoint_sweep(
        output_dir=args.output_dir,
        checkpoint_steps=parse_int_list(args.steps_list),
        dataset_path=args.dataset_path,
        loader_indices=parse_int_list(args.loader_indices),
        steps=args.steps,
        execution_horizon=args.execution_horizon,
        save_plots=args.save_plots,
    )
    write_json(args.output, summary)
    print_summary_table(summary)
    print(f"\nSummary: {args.output}")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
