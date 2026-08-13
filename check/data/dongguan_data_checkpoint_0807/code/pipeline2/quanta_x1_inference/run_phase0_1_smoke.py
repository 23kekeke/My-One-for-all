#!/usr/bin/env python3
"""Phase 0/1 smoke: parity check + open-loop on checkpoint-5000 val ep0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.constants import DEFAULT_CHECKPOINT, INFERENCE_TMP, VAL_DATASET
from quanta_x1_inference.open_loop import run_open_loop, write_json
from quanta_x1_inference.parity_check import run_parity_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Quanta X1 inference Phase 0/1 smoke.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", type=Path, default=VAL_DATASET)
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument(
        "--skip-open-loop",
        action="store_true",
        help="Only run parity checks (no GPU eval).",
    )
    args = parser.parse_args()

    INFERENCE_TMP.mkdir(parents=True, exist_ok=True)

    parity = run_parity_check(args.checkpoint, load_policy_flag=False)
    write_json(INFERENCE_TMP / "parity_check_processor.json", parity)
    print(f"Phase 0 processor parity: ok={parity['ok']}")

    from quanta_x1_inference.env import ensure_gr00t_imports

    ensure_gr00t_imports()
    parity_policy = run_parity_check(args.checkpoint, load_policy_flag=True)
    write_json(INFERENCE_TMP / "parity_check_policy.json", parity_policy)
    print(f"Phase 0 policy parity: ok={parity_policy['ok']}")

    if not parity["ok"] or not parity_policy["ok"]:
        raise SystemExit(1)

    if args.skip_open_loop:
        print("Skipped open-loop (--skip-open-loop).")
        return

    open_loop = run_open_loop(
        model_path=args.checkpoint,
        dataset_path=args.dataset_path,
        loader_index=args.loader_index,
        steps=args.steps,
        execution_horizon=args.execution_horizon,
        plot_path=INFERENCE_TMP / "plots/open_loop_ckpt5000_val.jpeg",
    )
    write_json(INFERENCE_TMP / "open_loop_ckpt5000_val.json", open_loop)
    print(
        f"Phase 1 open-loop: ok={open_loop['ok']} "
        f"mse={open_loop['mse']:.6f} mae={open_loop['mae']:.6f}"
    )
    if not open_loop["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
