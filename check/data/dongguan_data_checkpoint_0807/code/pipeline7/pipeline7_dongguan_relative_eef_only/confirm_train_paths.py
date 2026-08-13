#!/usr/bin/env python3
"""A2: confirm pipeline7 training paths, artifacts, and launcher wiring."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from manifest_utils import MULTI_DATASET, OUTPUT_ROOT, SMOKE_ROOT, write_json
from relative_train_utils import (
    BASE_MODEL,
    COSMOS_MODEL,
    GR00T_PYTHON,
    MULTI_DATASET as DATASET_DEFAULT,
    OUTPUT_ROOT as OUTPUT_DEFAULT,
    RELATIVE_MODALITY_CONFIG,
    check_relative_stats,
    load_dataset_meta,
)

PIPELINE7 = Path(__file__).resolve().parent
ROOT = PIPELINE7.parent

FINETUNE_ENTRY = PIPELINE7 / "dongguan_eef_only_finetune_entry.py"
LAUNCH_TRAIN = PIPELINE7 / "launch_train.sh"
RECOMMENDED_RUN = "dongguan_multi345_eef_only_relative_qlora_r32_vit_r32_bs2_ga4"
RECOMMENDED_OUTPUT = OUTPUT_DEFAULT / RECOMMENDED_RUN
PIPELINE5_OUTPUT = Path("/data/dongguan_data_checkpoint_0807/output")


def _writable_dir(path: Path) -> tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, "writable"
    except OSError as exc:
        return False, str(exc)


def _symlink_target(path: Path) -> str | None:
    if path.is_symlink():
        return str(path.resolve())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 A2 train path confirmation.")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=SMOKE_ROOT / "confirm_train_paths.json",
    )
    args = parser.parse_args()

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: dict | None = None) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail or {}})

    # Code + launcher
    train_files = {
        "finetune_entry": FINETUNE_ENTRY,
        "modality_config": RELATIVE_MODALITY_CONFIG,
        "launch_train": LAUNCH_TRAIN,
        "gr00t_hooks": PIPELINE7 / "gr00t_dongguan_hooks.py",
        "qlora_hooks": PIPELINE7 / "gr00t_qlora_hooks.py",
        "train_logging": PIPELINE7 / "train_logging_utils.py",
    }
    missing = [k for k, p in train_files.items() if not p.is_file()]
    add(
        "train_code_files",
        not missing,
        {"files": {k: str(v) for k, v in train_files.items()}, "missing": missing},
    )

    launch_exec = LAUNCH_TRAIN.is_file() and os.access(LAUNCH_TRAIN, os.X_OK)
    add("launch_train_executable", launch_exec, {"path": str(LAUNCH_TRAIN)})

    # Base models (must be GR00T-N1.7-3B, not pipeline5 ckpt)
    base_ok = BASE_MODEL.is_dir() and (BASE_MODEL / "config.json").is_file()
    add(
        "base_model_gr00t",
        base_ok,
        {"path": str(BASE_MODEL), "config_json": (BASE_MODEL / "config.json").is_file()},
    )

    cosmos_ok = COSMOS_MODEL.is_dir()
    add("cosmos_model", cosmos_ok, {"path": str(COSMOS_MODEL)})

    py_ok = GR00T_PYTHON.is_file() and os.access(GR00T_PYTHON, os.X_OK)
    add("gr00t_venv_python", py_ok, {"path": str(GR00T_PYTHON)})

    # Dataset
    ds = args.dataset_path
    meta_ok = (ds / "meta/info.json").is_file() and (ds / "meta/modality.json").is_file()
    add(
        "dataset_meta",
        meta_ok,
        {
            "dataset_path": str(ds.resolve()),
            "info_json": (ds / "meta/info.json").is_file(),
            "modality_json": (ds / "meta/modality.json").is_file(),
        },
    )

    if meta_ok:
        meta = load_dataset_meta(ds)
        add(
            "dataset_scale",
            int(meta.get("total_episodes", 0)) > 0 and int(meta.get("total_frames", 0)) > 0,
            {
                "total_episodes": meta.get("total_episodes"),
                "total_frames": meta.get("total_frames"),
                "fps": meta.get("fps"),
            },
        )

    rel = check_relative_stats(ds)
    add("relative_stats_eef_only", rel.get("ok", False), rel)

    sample_mp4 = ds / "videos/chunk-000/observation.images.head_camera/episode_000000.mp4"
    video_resolves = sample_mp4.exists()
    add(
        "videos_resolvable",
        video_resolves,
        {
            "sample": str(sample_mp4),
            "is_symlink": sample_mp4.is_symlink(),
            "target": _symlink_target(sample_mp4),
        },
    )

    # Output dirs
    out_ok, out_msg = _writable_dir(args.output_root)
    add("output_root_writable", out_ok, {"path": str(args.output_root.resolve()), "detail": out_msg})

    run_ok, run_msg = _writable_dir(RECOMMENDED_OUTPUT)
    add(
        "recommended_run_dir_writable",
        run_ok,
        {"path": str(RECOMMENDED_OUTPUT.resolve()), "run_name": RECOMMENDED_RUN, "detail": run_msg},
    )

    # launch_train.sh wiring
    launch_text = LAUNCH_TRAIN.read_text(encoding="utf-8") if LAUNCH_TRAIN.is_file() else ""
    wiring_ok = all(
        needle in launch_text
        for needle in (
            "lerobot_p7/multi_345",
            "output_p7",
            "dongguan_eef_only_finetune_entry.py",
            "dongguan_eef_only_relative_modality.py",
            RECOMMENDED_RUN,
            "GR00T-N1.7-3B",
        )
    )
    add(
        "launch_train_wiring",
        wiring_ok,
        {
            "dataset_needle": "lerobot_p7/multi_345",
            "output_needle": "output_p7",
            "run_name": RECOMMENDED_RUN,
        },
    )

    # Must NOT point base model at pipeline5 finetuned ckpt
    p5_ckpt_example = PIPELINE5_OUTPUT / "dongguan_multi345_relative_qlora_r32_vit_r32_bs2_ga4"
    not_p5_ckpt = "output/dongguan_multi345_relative" not in launch_text.replace("_p7", "")
    add(
        "base_not_pipeline5_checkpoint",
        not_p5_ckpt and "GR00T-N1.7-3B" in launch_text,
        {
            "launch_uses_gr00t_base": "GR00T-N1.7-3B" in launch_text,
            "pipeline5_ckpt_example": str(p5_ckpt_example),
            "note": "pipeline7 action_dim=20; cannot resume pipeline5 action_dim=32 ckpt",
        },
    )

    # Pre-train confirm report (if present)
    pre_report = SMOKE_ROOT / "pre_train_confirm.json"
    if pre_report.is_file():
        pre = json.loads(pre_report.read_text(encoding="utf-8"))
        add(
            "pre_train_confirm",
            bool(pre.get("ready_for_train")),
            {"path": str(pre_report), "ready_for_train": pre.get("ready_for_train")},
        )
    else:
        add("pre_train_confirm", False, {"reason": f"missing {pre_report}; run pre_train_confirm.py"})

    ok = all(c["ok"] for c in checks)
    report = {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "step": "confirm_train_paths",
        "ready_for_launch": ok,
        "checks": checks,
        "resolved_paths": {
            "dataset": str(ds.resolve()),
            "output_root": str(args.output_root.resolve()),
            "recommended_run_dir": str(RECOMMENDED_OUTPUT.resolve()),
            "base_model": str(BASE_MODEL.resolve()),
            "cosmos_model": str(COSMOS_MODEL.resolve()),
            "gr00t_python": str(GR00T_PYTHON.resolve()),
            "finetune_entry": str(FINETUNE_ENTRY.resolve()),
            "modality_config": str(RELATIVE_MODALITY_CONFIG.resolve()),
            "launch_train": str(LAUNCH_TRAIN.resolve()),
        },
        "launch_command": f"cd {PIPELINE7} && ./launch_train.sh multi345_v1",
    }
    write_json(args.output, report)

    print(f"Confirm train paths: ready_for_launch={ok}")
    for c in checks:
        mark = "OK" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}")
    print(f"  report: {args.output}")
    if ok:
        print(f"  launch: {report['launch_command']}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
