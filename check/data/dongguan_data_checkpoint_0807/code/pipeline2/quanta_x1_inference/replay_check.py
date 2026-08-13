"""Offline replay: verify observation builder matches LeRobot loader + open-loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.constants import DEFAULT_CHECKPOINT, INFERENCE_TMP, VAL_DATASET
from quanta_x1_inference.env import ensure_gr00t_imports
from quanta_x1_inference.observation import (
    LANGUAGE_KEY,
    build_flat_observation_from_state16,
    build_observation,
    build_observation_from_components,
    compare_flat_observations,
    compare_parsed_observations,
    flat_observation_from_step_data,
    inference_modality_configs,
    vector16_to_components,
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


def _concat_action(action_dict: dict[str, np.ndarray], action_keys: list[str], step: int = 0) -> np.ndarray:
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    parsed = parse_action_gr00t(action_dict)
    return np.concatenate(
        [np.atleast_1d(np.atleast_1d(parsed[f"action.{key}"])[step]) for key in action_keys],
        axis=0,
    )


def replay_one_step(
    *,
    dataset_path: Path | str,
    loader_index: int = 0,
    step_index: int = 0,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT,
    atol: float = 1e-5,
) -> dict[str, Any]:
    ensure_gr00t_imports()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    dataset_path = Path(dataset_path)
    modality_configs = get_registered_modality_configs()
    infer_configs = inference_modality_configs(modality_configs)

    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
    )
    if loader_index >= len(loader):
        raise IndexError(f"loader_index {loader_index} out of range")

    traj = loader[loader_index]
    ep_index = int(loader.episodes_metadata[loader_index]["episode_index"])
    step_data = extract_step_data(traj, step_index, infer_configs, EmbodimentTag.NEW_EMBODIMENT)

    loader_flat = flat_observation_from_step_data(step_data)

    head = np.asarray(step_data.images["head_camera"][0], dtype=np.uint8)
    wrist = np.asarray(step_data.images["right_arm_camera"][0], dtype=np.uint8)
    parts = {k: step_data.states[k] for k in step_data.states}

    builder_flat = build_observation_from_components(
        head_camera=head,
        right_arm_camera=wrist,
        eef_9d=parts["eef_9d"].reshape(-1),
        gripper_position=parts["gripper_position"].reshape(-1),
        joint_position=parts["joint_position"].reshape(-1),
        task_text=step_data.text,
        modality_configs=modality_configs,
    )[0]

    state_cols = [f"state.{k}" for k in modality_configs["state"].modality_keys]
    state_16d = np.concatenate(
        [np.asarray(traj[c].iloc[step_index], dtype=np.float32) for c in state_cols],
        axis=0,
    )
    vector_flat = build_flat_observation_from_state16(
        head_camera=head,
        right_arm_camera=wrist,
        state_16d=state_16d,
        task_text=step_data.text,
    )

    flat_cmp_builder = compare_flat_observations(loader_flat, builder_flat, atol=atol)
    flat_cmp_vector = compare_flat_observations(loader_flat, vector_flat, atol=atol)

    loader_parsed = build_observation(loader_flat, modality_configs)
    builder_parsed = build_observation(builder_flat, modality_configs)
    parsed_cmp = compare_parsed_observations(loader_parsed, builder_parsed, atol=atol)

    policy_report: dict[str, Any] | None = None
    if checkpoint_path is not None:
        policy = load_policy(checkpoint_path)
        action_keys = list(policy.modality_configs["action"].modality_keys)

        pred, _ = policy.get_action(loader_parsed)
        pred_action = _concat_action(pred, action_keys)

        gt_action = _ground_truth_action_vec(traj, step_index, action_keys)

        policy_report = {
            "checkpoint": str(checkpoint_path),
            "pred_action_finite": bool(np.all(np.isfinite(pred_action))),
            "pred_vs_gt_max_diff": float(np.max(np.abs(pred_action - gt_action))),
            "gt_action_l2": float(np.linalg.norm(gt_action)),
            "ok": bool(np.all(np.isfinite(pred_action))),
        }

    ok = (
        flat_cmp_builder["ok"]
        and flat_cmp_vector["ok"]
        and parsed_cmp["ok"]
        and (policy_report is None or policy_report["ok"])
    )

    return {
        "ok": ok,
        "dataset_path": str(dataset_path),
        "loader_index": loader_index,
        "step_index": step_index,
        "episode_index": ep_index,
        "language": step_data.text,
        "state_16d": state_16d.tolist(),
        "state_components": {k: vector16_to_components(state_16d)[k].tolist() for k in ("eef_9d",)},
        "flat_loader_vs_builder": flat_cmp_builder,
        "flat_loader_vs_vector16": flat_cmp_vector,
        "parsed_loader_vs_builder": parsed_cmp,
        "policy_single_step": policy_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 observation replay check.")
    parser.add_argument("--dataset-path", type=Path, default=VAL_DATASET)
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_TMP / "replay_one_step.json",
    )
    args = parser.parse_args()

    report = replay_one_step(
        dataset_path=args.dataset_path,
        loader_index=args.loader_index,
        step_index=args.step_index,
        checkpoint_path=None if args.skip_policy else args.checkpoint,
        atol=args.atol,
    )
    write_json(args.output, report)

    print(f"Replay one step: ok={report['ok']}")
    print(f"  ep_index={report['episode_index']} step={report['step_index']}")
    if report["policy_single_step"]:
        ps = report["policy_single_step"]
        print(
            f"  policy pred_finite={ps['pred_action_finite']} "
            f"pred_vs_gt={ps['pred_vs_gt_max_diff']:.3f}"
        )
    if report["flat_loader_vs_builder"]["errors"]:
        for err in report["flat_loader_vs_builder"]["errors"]:
            print(f"  ERROR: {err}")
    print(f"  report: {args.output}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
