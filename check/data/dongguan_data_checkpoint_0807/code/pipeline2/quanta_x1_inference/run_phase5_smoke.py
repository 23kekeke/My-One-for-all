#!/usr/bin/env python3
"""Phase 5 smoke: shadow live cycles (optional tiny live step with explicit flags)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.constants import INFERENCE_TMP  # noqa: E402
from quanta_x1_inference.live_capture import LIVE_ACK_TOKEN  # noqa: E402
from quanta_x1_inference.live_runner import build_parser, run_live_runner  # noqa: E402
from quanta_x1_inference.open_loop import write_json  # noqa: E402


def main() -> None:
    parser = build_parser()
    parser.description = "Phase 5 smoke: default 1 shadow cycle."
    parser.set_defaults(cycles=1, execution_horizon=1, interval_sec=0.0)
    parser.add_argument(
        "--live-one-step",
        action="store_true",
        help=(
            "Run one live cycle with execution_horizon=1. "
            f"Sets --mode live --execute and requires --acknowledge {LIVE_ACK_TOKEN}"
        ),
    )
    args = parser.parse_args()

    if args.live_one_step:
        args.mode = "live"
        args.execute = True
        args.cycles = 1
        args.execution_horizon = 1

    summary = run_live_runner(args)
    out = INFERENCE_TMP / "phase5_live_smoke.json"
    write_json(out, summary)
    print(f"Phase 5 smoke ok={summary['ok']}")
    print(f"  report: {out}")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
