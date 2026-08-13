#!/usr/bin/env python3
"""Phase 3 smoke: action decode vs open_loop_eval helper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.constants import DEFAULT_CHECKPOINT, DEFAULT_EXECUTION_HORIZON, INFERENCE_TMP, VAL_DATASET
from quanta_x1_inference.decode_check import run_action_decode_check
from quanta_x1_inference.open_loop import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Quanta X1 inference Phase 3 smoke.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", type=Path, default=VAL_DATASET)
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    args = parser.parse_args()

    report = run_action_decode_check(
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint,
        loader_index=args.loader_index,
        step_index=args.step_index,
        execution_horizon=args.execution_horizon,
    )
    out = INFERENCE_TMP / "phase3_action_decode.json"
    write_json(out, report)
    print(f"Phase 3 action decode: ok={report['ok']}")
    print(f"  report: {out}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
