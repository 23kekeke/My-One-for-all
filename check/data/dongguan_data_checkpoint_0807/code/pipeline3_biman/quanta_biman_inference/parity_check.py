"""Verify checkpoint processor contract matches Quanta biman 32D train settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quanta_biman_inference.constants import (
    DEFAULT_CHECKPOINT,
    EMBODIMENT_TAG_VALUE,
    EXPECTED_NEW_EMBODIMENT,
    EXPECTED_PROCESSOR_KWARGS,
    INFERENCE_TMP,
)
from quanta_biman_inference.env import ensure_gr00t_imports
from quanta_biman_inference.policy import load_policy, resolve_checkpoint


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


def load_model_config(checkpoint_dir: Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config.json under {checkpoint_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_keys(label: str, actual: list[str], expected: list[str]) -> dict[str, Any]:
    ok = actual == expected
    return {
        "name": label,
        "ok": ok,
        "expected": expected,
        "actual": actual,
    }


def _check_delta_indices(
    label: str,
    actual: list[int],
    expected: list[int],
) -> dict[str, Any]:
    ok = actual == expected
    return {
        "name": label,
        "ok": ok,
        "expected": expected,
        "actual": actual,
    }


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
        checks.append(
            {
                "name": "embodiment_present",
                "ok": False,
                "expected": EMBODIMENT_TAG_VALUE,
                "actual": list(modality_configs.keys()),
            }
        )
        errors.append(f"Missing embodiment {EMBODIMENT_TAG_VALUE!r} in processor modality_configs")
    else:
        emb = modality_configs[EMBODIMENT_TAG_VALUE]
        checks.append(
            _check_keys("video_keys", emb["video"]["modality_keys"], EXPECTED_NEW_EMBODIMENT["video_keys"])
        )
        checks.append(
            _check_keys("state_keys", emb["state"]["modality_keys"], EXPECTED_NEW_EMBODIMENT["state_keys"])
        )
        checks.append(
            _check_keys("action_keys", emb["action"]["modality_keys"], EXPECTED_NEW_EMBODIMENT["action_keys"])
        )
        checks.append(
            _check_keys(
                "language_keys",
                emb["language"]["modality_keys"],
                EXPECTED_NEW_EMBODIMENT["language_keys"],
            )
        )
        checks.append(
            _check_delta_indices(
                "video_delta_indices",
                list(emb["video"]["delta_indices"]),
                EXPECTED_NEW_EMBODIMENT["observation_delta_indices"],
            )
        )
        checks.append(
            _check_delta_indices(
                "state_delta_indices",
                list(emb["state"]["delta_indices"]),
                EXPECTED_NEW_EMBODIMENT["observation_delta_indices"],
            )
        )

        horizon = len(emb["action"]["delta_indices"])
        horizon_ok = horizon == EXPECTED_NEW_EMBODIMENT["action_horizon"]
        checks.append(
            {
                "name": "action_horizon",
                "ok": horizon_ok,
                "expected": EXPECTED_NEW_EMBODIMENT["action_horizon"],
                "actual": horizon,
            }
        )
        if not horizon_ok:
            errors.append(f"action horizon {horizon} != {EXPECTED_NEW_EMBODIMENT['action_horizon']}")

        action_reps = _normalize_action_reps(emb["action"].get("action_configs", []))
        reps_ok = action_reps == EXPECTED_NEW_EMBODIMENT["action_reps"]
        checks.append(
            {
                "name": "action_reps",
                "ok": reps_ok,
                "expected": EXPECTED_NEW_EMBODIMENT["action_reps"],
                "actual": action_reps,
            }
        )
        if not reps_ok:
            errors.append(f"action reps {action_reps} != {EXPECTED_NEW_EMBODIMENT['action_reps']}")

    for key, expected in EXPECTED_PROCESSOR_KWARGS.items():
        actual = kwargs.get(key)
        ok = actual == expected
        checks.append({"name": key, "ok": ok, "expected": expected, "actual": actual})
        if not ok:
            errors.append(f"{key}: {actual!r} != {expected!r}")

    use_relative_action = kwargs.get("use_relative_action")
    checks.append(
        {
            "name": "processor_use_relative_action",
            "ok": True,
            "actual": use_relative_action,
            "note": (
                "Model-level flag only; relative conversion runs only when "
                "action_configs[*].rep == RELATIVE AND use_relative_action is true. "
                "ABSOLUTE reps ignore this flag at decode time."
            ),
        }
    )
    if use_relative_action is True and EMBODIMENT_TAG_VALUE in modality_configs:
        emb = modality_configs[EMBODIMENT_TAG_VALUE]
        action_reps = _normalize_action_reps(emb["action"].get("action_configs", []))
        if any(rep != "ABSOLUTE" for rep in action_reps):
            checks.append(
                {
                    "name": "relative_conversion_would_activate",
                    "ok": False,
                    "action_reps": action_reps,
                }
            )
            errors.append(
                "processor use_relative_action=true with non-ABSOLUTE action reps; "
                "infer would apply relative→absolute conversion"
            )
        else:
            notes.append(
                "processor use_relative_action=true but all action reps are ABSOLUTE; "
                "decode uses absolute targets (same as pipeline2)."
            )

    model_cfg = load_model_config(checkpoint_dir)
    state_hist = int(model_cfg.get("state_history_length", 1))
    state_hist_ok = state_hist == EXPECTED_NEW_EMBODIMENT["state_history_length"]
    checks.append(
        {
            "name": "model_state_history_length",
            "ok": state_hist_ok,
            "expected": EXPECTED_NEW_EMBODIMENT["state_history_length"],
            "actual": state_hist,
        }
    )
    if not state_hist_ok:
        errors.append(
            f"model state_history_length {state_hist} != "
            f"{EXPECTED_NEW_EMBODIMENT['state_history_length']}"
        )

    stats_path = checkpoint_dir / "statistics.json"
    stats_ok = stats_path.is_file()
    checks.append({"name": "statistics_json", "ok": stats_ok, "path": str(stats_path)})
    if stats_ok:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        has_emb = EMBODIMENT_TAG_VALUE in stats
        checks.append(
            {
                "name": "statistics_embodiment",
                "ok": has_emb,
                "expected": EMBODIMENT_TAG_VALUE,
                "actual": list(stats.keys())[:8],
            }
        )
        if not has_emb:
            errors.append(f"statistics.json missing {EMBODIMENT_TAG_VALUE!r}")
    else:
        errors.append(f"Missing {stats_path}")

    ok = all(c.get("ok", True) for c in checks) and not errors
    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "ok": ok,
        "checks": checks,
        "errors": errors,
        "notes": notes,
    }


def check_loaded_policy(checkpoint_dir: Path) -> dict[str, Any]:
    """Load policy and verify modality configs match checkpoint processor."""
    checkpoint_dir = Path(checkpoint_dir)
    policy = load_policy(checkpoint_dir)
    proc_report = check_processor_contract(checkpoint_dir)
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
            exp_delta = EXPECTED_NEW_EMBODIMENT["observation_delta_indices"]
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
    sh_ok = model_state_hist == EXPECTED_NEW_EMBODIMENT["state_history_length"]
    policy_checks.append(
        {
            "name": "policy_model_state_history_length",
            "ok": sh_ok,
            "expected": EXPECTED_NEW_EMBODIMENT["state_history_length"],
            "actual": model_state_hist,
        }
    )
    if not sh_ok:
        errors.append(
            f"loaded model state_history_length={model_state_hist}, "
            f"expected {EXPECTED_NEW_EMBODIMENT['state_history_length']}"
        )

    ok = proc_report["ok"] and all(c["ok"] for c in policy_checks) and not errors
    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "ok": ok,
        "processor_checks": proc_report["checks"],
        "policy_checks": policy_checks,
        "errors": errors,
        "notes": notes,
    }


def run_parity_check(
    checkpoint_dir: Path | str | None = None,
    *,
    load_policy_flag: bool = False,
) -> dict[str, Any]:
    checkpoint_dir = resolve_checkpoint(Path(checkpoint_dir or DEFAULT_CHECKPOINT))
    if checkpoint_dir is None:
        raise FileNotFoundError("No checkpoint directory resolved")
    if load_policy_flag:
        return check_loaded_policy(checkpoint_dir)
    return check_processor_contract(checkpoint_dir)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quanta biman 32D train/inference parity check.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Finetuned checkpoint dir (default: task1 checkpoint-15000).",
    )
    parser.add_argument(
        "--load-policy",
        action="store_true",
        help="Also load QLoRA policy and verify runtime modality configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_TMP / "parity_check.json",
    )
    args = parser.parse_args()

    if args.load_policy:
        ensure_gr00t_imports()
    report = run_parity_check(args.checkpoint, load_policy_flag=args.load_policy)
    write_json(args.output, report)

    print(f"Parity check ok={report['ok']}")
    for note in report.get("notes", []):
        print(f"  NOTE: {note}")
    if report["errors"]:
        for err in report["errors"]:
            print(f"  ERROR: {err}")
    print(f"  report: {args.output}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
