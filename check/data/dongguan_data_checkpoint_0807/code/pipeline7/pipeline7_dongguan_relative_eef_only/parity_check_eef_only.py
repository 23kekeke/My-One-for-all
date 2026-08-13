#!/usr/bin/env python3
"""Verify Dongguan pipeline7 eef-only checkpoint processor + optional policy load."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dongguan_eef_inference.constants import (
    DEFAULT_CHECKPOINT,
    EXPECTED_ACTION_REPS,
    EXPECTED_NEW_EMBODIMENT,
    INFERENCE_TMP,
    OBSERVATION_DELTA_INDICES,
    STATE_HISTORY_LENGTH,
)
from dongguan_eef_inference.env import ensure_dongguan_infer_imports
from dongguan_eef_inference.patch import apply_infer_patches
from dongguan_eef_inference.policy import load_policy, resolve_checkpoint
from relative_train_utils import EMBODIMENT_TAG_VALUE, write_json

VIDEO_KEYS = EXPECTED_NEW_EMBODIMENT["video_keys"]
STATE_KEYS = EXPECTED_NEW_EMBODIMENT["state_keys"]
ACTION_KEYS = EXPECTED_NEW_EMBODIMENT["action_keys"]


def _processor_dir(checkpoint_dir: Path) -> Path:
    nested = checkpoint_dir / "processor"
    if nested.is_dir() and (nested / "processor_config.json").is_file():
        return nested
    return checkpoint_dir


def load_processor_config(checkpoint_dir: Path) -> dict[str, Any]:
    path = _processor_dir(checkpoint_dir) / "processor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing processor_config.json under {checkpoint_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_action_reps(raw: list[Any]) -> list[str]:
    reps: list[str] = []
    for cfg in raw:
        if isinstance(cfg, dict):
            rep = cfg.get("rep", "")
            reps.append(str(rep).split(".")[-1].upper())
        else:
            reps.append(str(cfg).upper())
    return reps


def check_processor_contract(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    notes: list[str] = []

    proc_cfg = load_processor_config(checkpoint_dir)
    kwargs = proc_cfg.get("processor_kwargs", proc_cfg)
    modality_configs = kwargs.get("modality_configs", {})

    if EMBODIMENT_TAG_VALUE not in modality_configs:
        errors.append(f"Missing embodiment {EMBODIMENT_TAG_VALUE!r}")
    else:
        emb = modality_configs[EMBODIMENT_TAG_VALUE]
        action_keys = list(emb["action"]["modality_keys"])
        keys_ok = action_keys == list(ACTION_KEYS)
        checks.append(
            {
                "name": "action_keys",
                "ok": keys_ok,
                "expected": list(ACTION_KEYS),
                "actual": action_keys,
            }
        )
        if not keys_ok:
            errors.append(f"action keys {action_keys} != {list(ACTION_KEYS)}")

        action_reps = _normalize_action_reps(emb["action"].get("action_configs", []))
        reps_ok = action_reps == EXPECTED_ACTION_REPS
        checks.append(
            {
                "name": "action_reps",
                "ok": reps_ok,
                "expected": EXPECTED_ACTION_REPS,
                "actual": action_reps,
            }
        )
        if not reps_ok:
            errors.append(f"action reps {action_reps} != {EXPECTED_ACTION_REPS}")

        video_deltas = list(emb["video"]["delta_indices"])
        state_deltas = list(emb["state"]["delta_indices"])
        checks.append(
            {
                "name": "video_delta_indices",
                "ok": video_deltas == OBSERVATION_DELTA_INDICES,
                "expected": OBSERVATION_DELTA_INDICES,
                "actual": video_deltas,
            }
        )
        checks.append(
            {
                "name": "state_delta_indices",
                "ok": state_deltas == OBSERVATION_DELTA_INDICES,
                "expected": OBSERVATION_DELTA_INDICES,
                "actual": state_deltas,
            }
        )
        if video_deltas != OBSERVATION_DELTA_INDICES:
            errors.append(f"video delta_indices mismatch: {video_deltas}")
        if state_deltas != OBSERVATION_DELTA_INDICES:
            errors.append(f"state delta_indices mismatch: {state_deltas}")

        state_keys = list(emb["state"]["modality_keys"])
        state_ok = state_keys == list(STATE_KEYS)
        checks.append(
            {
                "name": "state_keys",
                "ok": state_ok,
                "expected": list(STATE_KEYS),
                "actual": state_keys,
            }
        )
        if not state_ok:
            errors.append(f"state keys {state_keys} != {list(STATE_KEYS)}")

    use_relative_action = kwargs.get("use_relative_action")
    rel_flag_ok = use_relative_action is True
    checks.append(
        {
            "name": "use_relative_action",
            "ok": rel_flag_ok,
            "expected": True,
            "actual": use_relative_action,
        }
    )
    if not rel_flag_ok:
        errors.append(f"use_relative_action={use_relative_action!r}, expected True")

    model_cfg_path = checkpoint_dir / "config.json"
    if model_cfg_path.is_file():
        model_cfg = json.loads(model_cfg_path.read_text(encoding="utf-8"))
        state_hist = int(model_cfg.get("state_history_length", 1))
        hist_ok = state_hist == STATE_HISTORY_LENGTH
        checks.append(
            {
                "name": "state_history_length",
                "ok": hist_ok,
                "expected": STATE_HISTORY_LENGTH,
                "actual": state_hist,
            }
        )
        if not hist_ok:
            errors.append(f"state_history_length {state_hist} != {STATE_HISTORY_LENGTH}")

    stats_path = checkpoint_dir / "statistics.json"
    stats_ok = stats_path.is_file()
    checks.append({"name": "statistics_json", "ok": stats_ok})
    if not stats_ok:
        errors.append(f"Missing {stats_path}")
    else:
        notes.append(
            "Infer: use dongguan_eef_inference bootstrap (see infer_dongguan_eef_only.txt); "
            "do not load with pipeline5 32D-action modality."
        )

    ok = not errors and all(c.get("ok", True) for c in checks)
    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "ok": ok,
        "checks": checks,
        "errors": errors,
        "notes": notes,
    }


def check_loaded_policy(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    ensure_dongguan_infer_imports()
    apply_infer_patches()

    proc_report = check_processor_contract(checkpoint_dir)
    policy = load_policy(checkpoint_dir)
    proc_cfg = load_processor_config(checkpoint_dir)
    kwargs = proc_cfg.get("processor_kwargs", proc_cfg)
    expected_emb = kwargs["modality_configs"][EMBODIMENT_TAG_VALUE]

    policy_checks: list[dict[str, Any]] = []
    errors = list(proc_report["errors"])
    notes = list(proc_report.get("notes", []))

    for modality in ("video", "state", "action", "language"):
        cfg = policy.modality_configs[modality]
        exp_keys = expected_emb[modality]["modality_keys"]
        act_keys = list(cfg.modality_keys)
        ok = act_keys == exp_keys
        policy_checks.append(
            {"name": f"policy_{modality}_keys", "ok": ok, "expected": exp_keys, "actual": act_keys}
        )
        if not ok:
            errors.append(f"policy {modality} keys {act_keys} != {exp_keys}")

        if modality in ("video", "state"):
            exp_delta = OBSERVATION_DELTA_INDICES
            act_delta = list(cfg.delta_indices)
            delta_ok = act_delta == exp_delta
            policy_checks.append(
                {
                    "name": f"policy_{modality}_delta_indices",
                    "ok": delta_ok,
                    "expected": exp_delta,
                    "actual": act_delta,
                }
            )
            if not delta_ok:
                errors.append(f"policy {modality} delta_indices {act_delta} != {exp_delta}")

    letterbox = getattr(policy.processor, "letter_box_transform", None)
    lb_ok = letterbox is True
    policy_checks.append(
        {
            "name": "policy_letter_box_transform",
            "ok": lb_ok,
            "expected": True,
            "actual": letterbox,
        }
    )
    if not lb_ok:
        errors.append(f"processor.letter_box_transform={letterbox!r}, expected True")

    model_state_hist = int(getattr(policy.model.config, "state_history_length", 1))
    sh_ok = model_state_hist == STATE_HISTORY_LENGTH
    policy_checks.append(
        {
            "name": "policy_model_state_history_length",
            "ok": sh_ok,
            "expected": STATE_HISTORY_LENGTH,
            "actual": model_state_hist,
        }
    )
    if not sh_ok:
        errors.append(
            f"loaded model state_history_length={model_state_hist}, "
            f"expected {STATE_HISTORY_LENGTH}"
        )

    ok = proc_report["ok"] and all(c["ok"] for c in policy_checks) and not errors
    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "ok": ok,
        "processor_checks": proc_report["checks"],
        "policy_checks": policy_checks,
        "errors": errors,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 Dongguan eef-only parity check.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--load-policy",
        action="store_true",
        help="Also load QLoRA policy on GPU and verify runtime modality configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.load_policy:
        report = check_loaded_policy(checkpoint)
    else:
        report = check_processor_contract(checkpoint)

    print(f"Dongguan eef-only parity: ok={report['ok']}")
    for key in ("checks", "processor_checks"):
        for check in report.get(key, []):
            print(f"  {check['name']}: ok={check.get('ok')} actual={check.get('actual')}")
    for check in report.get("policy_checks", []):
        print(f"  {check['name']}: ok={check.get('ok')} actual={check.get('actual')}")
    for err in report["errors"]:
        print(f"  ERROR: {err}")
    for note in report.get("notes", []):
        print(f"  note: {note}")

    output = args.output
    if output is None:
        suffix = "policy" if args.load_policy else "processor"
        output = INFERENCE_TMP / f"parity_check_{suffix}.json"
    write_json(output, report)
    print(f"Report: {output}")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
