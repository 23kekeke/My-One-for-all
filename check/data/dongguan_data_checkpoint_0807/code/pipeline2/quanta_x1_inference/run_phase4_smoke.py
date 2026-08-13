#!/usr/bin/env python3
"""Phase 4 smoke: val open-loop sweep on default checkpoint steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.checkpoint_sweep import (
    print_summary_table,
    run_checkpoint_sweep,
)
from quanta_x1_inference.constants import DEFAULT_SWEEP_CHECKPOINT_STEPS, DEFAULT_TRAIN_OUTPUT, INFERENCE_TMP
from quanta_x1_inference.open_loop import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 checkpoint sweep smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument(
        "--steps-list",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SWEEP_CHECKPOINT_STEPS),
    )
    parser.add_argument("--loader-indices", type=str, default="0")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    from quanta_x1_inference.checkpoint_sweep import parse_int_list

    summary = run_checkpoint_sweep(
        output_dir=args.output_dir,
        checkpoint_steps=parse_int_list(args.steps_list),
        dataset_path="val",
        loader_indices=parse_int_list(args.loader_indices),
        save_plots=args.save_plots,
    )
    out = INFERENCE_TMP / "phase4_sweep_summary.json"
    write_json(out, summary)
    print_summary_table(summary)
    print(f"\nPhase 4 sweep ok={summary['ok']}")
    print(f"  report: {out}")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
