#!/usr/bin/env python3
"""Phase 2 smoke: observation replay vs LeRobot loader (+ optional policy step)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.constants import DEFAULT_CHECKPOINT, INFERENCE_TMP, VAL_DATASET
from quanta_x1_inference.open_loop import write_json
from quanta_x1_inference.replay_check import replay_one_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 observation replay smoke.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", type=Path, default=VAL_DATASET)
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--skip-policy", action="store_true")
    args = parser.parse_args()

    report = replay_one_step(
        dataset_path=args.dataset_path,
        loader_index=args.loader_index,
        step_index=args.step_index,
        checkpoint_path=None if args.skip_policy else args.checkpoint,
    )
    out = INFERENCE_TMP / "phase2_replay_one_step.json"
    write_json(out, report)
    print(f"Phase 2 replay: ok={report['ok']}")
    print(f"  report: {out}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
