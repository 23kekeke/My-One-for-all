#!/usr/bin/env python3
"""Pre-train confirmation gate for pipeline7 (after Step4–6, before finetune)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relative_train_utils import (
    ACTION_DIM,
    EXPECTED_ACTION_REPS,
    MULTI_DATASET,
    RELATIVE_ACTION_KEYS,
    RELATIVE_MODALITY_CONFIG,
    SMOKE_ROOT,
    STATE_DIM,
    check_relative_stats,
    ensure_gr00t_imports,
    get_registered_modality_configs,
    load_dataset_meta,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 pre-train confirmation.")
    parser.add_argument("--dataset-path", type=Path, default=MULTI_DATASET)
    parser.add_argument(
        "--step4-report",
        type=Path,
        default=MULTI_DATASET / "export_report.json",
    )
    parser.add_argument(
        "--step5-report",
        type=Path,
        default=SMOKE_ROOT / "multi_345_relative_stats_report.json",
    )
    parser.add_argument(
        "--step6-report",
        type=Path,
        default=SMOKE_ROOT / "multi_345_loader_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SMOKE_ROOT / "pre_train_confirm.json",
    )
    args = parser.parse_args()

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: dict | None = None) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail or {}})

    # Step4 export report
    if args.step4_report.is_file():
        s4 = json.loads(args.step4_report.read_text(encoding="utf-8"))
        add(
            "step4_export_ok",
            s4.get("validation_ok") == s4.get("exported_episodes")
            and s4.get("state_dim") == STATE_DIM
            and s4.get("action_dim") == ACTION_DIM
            and not s4.get("failures"),
            {
                "exported": s4.get("exported_episodes"),
                "validation_ok": s4.get("validation_ok"),
                "link_videos_from": s4.get("link_videos_from"),
            },
        )
    else:
        add("step4_export_ok", False, {"reason": f"missing {args.step4_report}"})

    # Step5
    if args.step5_report.is_file():
        s5 = json.loads(args.step5_report.read_text(encoding="utf-8"))
        add("step5_relative_stats_ok", bool(s5.get("ok")), s5.get("relative_stats_check"))
    else:
        add("step5_relative_stats_ok", False, {"reason": f"missing {args.step5_report}"})

    # Step6
    if args.step6_report.is_file():
        s6 = json.loads(args.step6_report.read_text(encoding="utf-8"))
        add("step6_loader_smoke_ok", bool(s6.get("ok")), {"action_reps": s6.get("actual_action_reps")})
    else:
        add("step6_loader_smoke_ok", False, {"reason": f"missing {args.step6_report}"})

    # Dataset meta
    meta = load_dataset_meta(args.dataset_path)
    features = meta.get("features", {})
    add(
        "info_shapes",
        features.get("observation.state", {}).get("shape") == [STATE_DIM]
        and features.get("action", {}).get("shape") == [ACTION_DIM],
        {
            "state": features.get("observation.state", {}).get("shape"),
            "action": features.get("action", {}).get("shape"),
        },
    )

    rel = check_relative_stats(args.dataset_path)
    add(
        "relative_stats_keys",
        rel.get("ok") and set(RELATIVE_ACTION_KEYS).issubset(set(rel.get("keys_present", []))),
        rel,
    )

    ensure_gr00t_imports()
    modality = get_registered_modality_configs()
    add(
        "modality_config",
        len(modality["state"].modality_keys) == 6
        and len(modality["action"].modality_keys) == 4
        and [c.rep.name for c in modality["action"].action_configs] == EXPECTED_ACTION_REPS,
        {
            "state_keys": modality["state"].modality_keys,
            "action_keys": modality["action"].modality_keys,
            "action_reps": [c.rep.name for c in modality["action"].action_configs],
            "modality_config_path": str(RELATIVE_MODALITY_CONFIG),
        },
    )

    # Video symlinks sample
    video_root = args.dataset_path / "videos/chunk-000/observation.images.head_camera"
    sample = video_root / "episode_000000.mp4"
    video_ok = sample.is_symlink() or sample.is_file()
    add(
        "videos_present",
        video_ok,
        {"sample": str(sample), "is_symlink": sample.is_symlink() if sample.exists() else False},
    )

    ok = all(c["ok"] for c in checks)
    report = {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "step": "pre_train_confirm",
        "dataset_path": str(args.dataset_path.resolve()),
        "ready_for_train": ok,
        "checks": checks,
        "train_contract": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_keys": modality["action"].modality_keys,
            "relative_keys": list(RELATIVE_ACTION_KEYS),
            "delta_indices": [-7, -1, 0],
            "use_relative_action": True,
            "base_model": "GR00T-N1.7-3B",
            "recommended_run_name": "dongguan_multi345_eef_only_relative_qlora_r32_vit_r32_bs2_ga4",
        },
    }
    write_json(args.output, report)

    print(f"Pre-train confirm: ready_for_train={ok}")
    for c in checks:
        mark = "OK" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}")
    print(f"  report: {args.output}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
