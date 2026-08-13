"""Offline checks that action_decode matches open_loop_eval slicing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.action_decode import (
    decode_action_at_step,
    decode_open_loop_chunk,
    open_loop_concat_at_step,
    validate_execution_horizon,
)
from quanta_x1_inference.constants import DEFAULT_CHECKPOINT, DEFAULT_EXECUTION_HORIZON, INFERENCE_TMP, VAL_DATASET
from quanta_x1_inference.env import ensure_gr00t_imports
from quanta_x1_inference.observation import (
    build_observation,
    flat_observation_from_step_data,
    inference_modality_configs,
)
from quanta_x1_inference.open_loop import get_registered_modality_configs, write_json
from quanta_x1_inference.policy import load_policy


def _ground_truth_action_vec(traj, step_index: int, action_keys: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in action_keys:
        val = np.asarray(traj[f"action.{key}"].iloc[step_index], dtype=np.float32)
        if val.ndim == 2:
            val = val[0]
        parts.append(val.reshape(-1))
    return np.concatenate(parts, axis=0)


def run_action_decode_check(
    *,
    dataset_path: Path | str,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    loader_index: int = 0,
    step_index: int = 0,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
) -> dict[str, Any]:
    ensure_gr00t_imports()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    dataset_path = Path(dataset_path)
    modality_configs = get_registered_modality_configs()
    infer_configs = inference_modality_configs(modality_configs)
    validate_execution_horizon(modality_configs, execution_horizon)

    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
    )
    traj = loader[loader_index]
    ep_index = int(loader.episodes_metadata[loader_index]["episode_index"])
    step_data = extract_step_data(traj, step_index, infer_configs, EmbodimentTag.NEW_EMBODIMENT)

    policy = load_policy(checkpoint_path)
    action_keys = list(policy.modality_configs["action"].modality_keys)

    flat_obs = flat_observation_from_step_data(step_data)
    parsed_obs = build_observation(flat_obs, modality_configs)
    action_dict, _ = policy.get_action(parsed_obs)

    per_step_checks: list[dict[str, Any]] = []
    decode_errors: list[str] = []

    for j in range(execution_horizon):
        ours = open_loop_concat_at_step(action_dict, j, action_keys_list=action_keys)
        parsed = parse_action_gr00t(action_dict)
        ref = np.concatenate(
            [np.atleast_1d(parsed[f"action.{key}"][j]) for key in action_keys],
            axis=0,
        ).astype(np.float32)
        diff = float(np.max(np.abs(ours - ref)))
        step_ok = bool(diff <= 1e-6 and np.all(np.isfinite(ours)))
        per_step_checks.append({"step": j, "ok": step_ok, "max_abs_diff": diff})
        if not step_ok:
            decode_errors.append(f"step {j}: max_abs_diff={diff:.3e}")

    chunk = decode_open_loop_chunk(
        action_dict,
        execution_horizon,
        action_keys_list=action_keys,
    )
    ref_chunk = np.stack(
        [open_loop_concat_at_step(action_dict, j, action_keys_list=action_keys) for j in range(execution_horizon)],
        axis=0,
    )
    chunk_ok = bool(np.allclose(chunk, ref_chunk))

    decoded0 = decode_action_at_step(action_dict, 0, action_keys_list=action_keys)
    gt0 = _ground_truth_action_vec(traj, step_index, action_keys)
    gt_diff = float(np.max(np.abs(decoded0.vector16 - gt0)))

    ok = bool(chunk_ok and all(c["ok"] for c in per_step_checks) and np.all(np.isfinite(chunk)))
    return {
        "ok": ok,
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "loader_index": loader_index,
        "step_index": step_index,
        "episode_index": ep_index,
        "execution_horizon": execution_horizon,
        "per_step_checks": per_step_checks,
        "chunk_shape": list(chunk.shape),
        "chunk_matches_open_loop_helper": chunk_ok,
        "decoded_step0_vs_gt_max_diff": gt_diff,
        "decoded_step0_sdk_joints": decoded0.sdk_joint_targets.tolist(),
        "errors": decode_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 action decode check.")
    parser.add_argument("--dataset-path", type=Path, default=VAL_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_TMP / "action_decode_check.json",
    )
    args = parser.parse_args()

    report = run_action_decode_check(
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint,
        loader_index=args.loader_index,
        step_index=args.step_index,
        execution_horizon=args.execution_horizon,
    )
    write_json(args.output, report)

    print(f"Action decode check: ok={report['ok']}")
    print(f"  ep_index={report['episode_index']} step={report['step_index']}")
    print(f"  step0 vs gt max_diff={report['decoded_step0_vs_gt_max_diff']:.3f}")
    print(f"  report: {args.output}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
