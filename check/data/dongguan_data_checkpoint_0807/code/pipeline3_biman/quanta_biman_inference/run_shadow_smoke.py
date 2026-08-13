#!/usr/bin/env python3
"""Shadow smoke: one offline val infer + optional one live shadow cycle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE3 = Path(__file__).resolve().parents[1]
if str(_PIPELINE3) not in sys.path:
    sys.path.insert(0, str(_PIPELINE3))

from quanta_biman_inference.constants import DEFAULT_CHECKPOINT, INFERENCE_TMP, TASK1_DATASET
from quanta_biman_inference.live_runner import build_parser as build_live_parser, run_live_runner
from quanta_biman_inference.offline_infer import infer_offline_episode, write_json
from quanta_biman_inference.policy import load_policy, resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Biman inference smoke test.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-live-shadow", action="store_true")
    parser.add_argument("--task-index", type=int, default=0)
    args, _rest = parser.parse_known_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    summary: dict = {"checkpoint": str(checkpoint), "ok": True}

    if not args.skip_offline:
        print("=== Offline val infer (no robot) ===")
        policy = load_policy(checkpoint)
        offline = infer_offline_episode(
            policy=policy,
            dataset_path=TASK1_DATASET,
            loader_index=0,
            step_index=0,
            execution_horizon=1,
        )
        offline["model_path"] = str(checkpoint)
        out = INFERENCE_TMP / "smoke_offline_infer.json"
        write_json(out, offline)
        summary["offline"] = {"ok": offline["ok"], "report": str(out)}
        print(f"  task_index={offline['task_index']} report={out}")

    if not args.skip_live_shadow:
        print("=== Live shadow (requires robot SDK tunnel) ===")
        live_parser = build_live_parser()
        live_args = live_parser.parse_args(
            [
                "--mode",
                "shadow",
                "--cycles",
                "1",
                "--execution-horizon",
                "1",
                "--task-index",
                str(args.task_index),
                "--checkpoint",
                str(checkpoint),
            ]
        )
        live_summary = run_live_runner(live_args)
        summary["live_shadow"] = live_summary
        print(f"  run_root={live_summary['run_root']}")

    out = INFERENCE_TMP / "smoke_summary.json"
    write_json(out, summary)
    print(f"\nSmoke summary: {out}")


if __name__ == "__main__":
    main()
