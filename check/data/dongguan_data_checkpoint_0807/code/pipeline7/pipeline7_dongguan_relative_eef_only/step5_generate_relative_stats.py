#!/usr/bin/env python3
"""Generate stats.json + relative_stats.json for pipeline7 RELATIVE eef-only training."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from manifest_utils import SMOKE_ROOT
from relative_train_utils import (
    MULTI_DATASET,
    RELATIVE_MODALITY_CONFIG,
    check_relative_stats,
    ensure_gr00t_imports,
    setup_gr00t_env,
    write_json,
)


def generate_stats(dataset_path: Path) -> dict:
    ensure_gr00t_imports()
    importlib.import_module("dongguan_eef_only_relative_config")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import main as stats_main

    stats_main(
        dataset_path,
        EmbodimentTag.NEW_EMBODIMENT,
        modality_config_path=str(RELATIVE_MODALITY_CONFIG),
    )

    rel_check = check_relative_stats(dataset_path)
    stats_path = dataset_path / "meta/stats.json"
    rel_path = dataset_path / "meta/relative_stats.json"
    return {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "step": "step5_generate_relative_stats",
        "dataset_path": str(dataset_path.resolve()),
        "stats_json": str(stats_path),
        "relative_stats_json": str(rel_path),
        "relative_stats_check": rel_check,
        "ok": rel_check["ok"] and stats_path.is_file() and rel_path.is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 relative stats generation.")
    parser.add_argument("--dataset-path", type=Path, default=MULTI_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=SMOKE_ROOT / "multi_345_relative_stats_report.json",
    )
    args = parser.parse_args()

    if not args.dataset_path.is_dir():
        raise FileNotFoundError(args.dataset_path)

    env = setup_gr00t_env()
    for key, value in env.items():
        os.environ[key] = value

    report = generate_stats(args.dataset_path)
    write_json(args.output, report)

    print(f"Step5 relative stats: ok={report['ok']}")
    print(f"  dataset: {report['dataset_path']}")
    print(f"  relative_stats keys: {report['relative_stats_check'].get('keys_present')}")
    if report["relative_stats_check"].get("missing_keys"):
        print(f"  missing: {report['relative_stats_check']['missing_keys']}")
    print(f"  report: {args.output}")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
