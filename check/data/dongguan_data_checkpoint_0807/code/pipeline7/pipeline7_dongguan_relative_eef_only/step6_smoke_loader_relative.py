#!/usr/bin/env python3
"""Loader smoke for pipeline7 RELATIVE eef-only (state 32D / action 20D)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dongguan_eef_only_relative_config import OBSERVATION_DELTA_INDICES, STATE_HISTORY_LENGTH
from relative_train_utils import (
    ACTION_DIM,
    EXPECTED_ACTION_REPS,
    MULTI_DATASET,
    SMOKE_ROOT,
    STATE_DIM,
    check_relative_stats,
    ensure_gr00t_imports,
    get_registered_modality_configs,
    load_dataset_meta,
    load_tasks_jsonl,
    write_json,
)


def check_action_semantics(traj, state_keys: list[str], action_keys: list[str], tol: float = 1e-4) -> dict:
    state_key_set = set(state_keys)
    diffs: list[float] = []
    per_key: dict[str, float] = {}
    for key in action_keys:
        if key not in state_key_set:
            continue
        state = np.vstack(traj[f"state.{key}"].to_numpy())
        action = np.vstack(traj[f"action.{key}"].to_numpy())
        key_diffs = [
            float(np.max(np.abs(action[t] - state[t + 1]))) for t in range(len(traj) - 1)
        ]
        per_key[key] = max(key_diffs) if key_diffs else 0.0
        diffs.extend(key_diffs)
    max_diff = max(diffs) if diffs else 0.0
    return {
        "frames_checked": len(traj) - 1,
        "max_abs_diff": max_diff,
        "per_key_max_abs_diff": per_key,
        "ok": max_diff <= tol,
    }


def check_state_history_shapes(states: dict[str, np.ndarray]) -> dict:
    checks = {}
    for key, arr in states.items():
        shape_ok = arr.ndim == 2 and arr.shape[0] == STATE_HISTORY_LENGTH
        checks[key] = {"shape": list(arr.shape), "ok": shape_ok}
    return {
        "ok": all(v["ok"] for v in checks.values()),
        "groups": checks,
    }


def check_relative_processor_conversion(
    *,
    raw_state: dict[str, np.ndarray],
    raw_action: dict[str, np.ndarray],
    step_index: int,
) -> dict:
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.types import ActionFormat, ActionType

    ensure_gr00t_imports()

    processor = StateActionProcessor(
        modality_configs={},
        statistics={},
        use_relative_action=True,
    )

    checks = {}
    for eef_key in ("left_eef_9d", "right_eef_9d"):
        rel_eef = processor._convert_to_relative_action(
            raw_action[eef_key],
            raw_state[eef_key][-1],
            ActionType.EEF,
            ActionFormat.XYZ_ROT6D,
        )
        eef_max = float(np.max(np.abs(rel_eef)))
        checks[eef_key] = {
            "max_abs_relative": eef_max,
            "nonzero_relative": eef_max > 1e-6,
        }

    ok = (
        checks["right_eef_9d"]["nonzero_relative"]
        and (
            checks["left_eef_9d"]["nonzero_relative"]
            or checks["left_eef_9d"]["max_abs_relative"] < 1e-3
        )
    )
    return {"ok": ok, "step_index": step_index, "groups": checks}


def run_loader_smoke(
    dataset_path: Path,
    *,
    loader_indices: tuple[int, ...] | None = None,
    step_index: int = 50,
) -> dict:
    ensure_gr00t_imports()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
        extract_step_data,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag

    modality_configs = get_registered_modality_configs()
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
    )

    info = load_dataset_meta(dataset_path)
    tasks_map = load_tasks_jsonl(dataset_path)
    num_episodes = len(loader)
    if loader_indices is None:
        loader_indices = (0,) if num_episodes <= 1 else (0, num_episodes // 2, num_episodes - 1)

    rel_stats_check = check_relative_stats(dataset_path)

    action_configs = modality_configs["action"].action_configs
    action_reps = [cfg.rep.name for cfg in action_configs]

    delta_ok = list(OBSERVATION_DELTA_INDICES) == [-7, -1, 0]

    state_shape = info.get("features", {}).get("observation.state", {}).get("shape")
    action_shape = info.get("features", {}).get("action", {}).get("shape")

    report: dict = {
        "pipeline": "pipeline7_dongguan_relative_eef_only",
        "step": "step6_smoke_loader_relative",
        "dataset_path": str(dataset_path.resolve()),
        "num_episodes": num_episodes,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "observation_delta_indices": OBSERVATION_DELTA_INDICES,
        "state_history_length": STATE_HISTORY_LENGTH,
        "delta_indices_ok": delta_ok,
        "expected_action_reps": EXPECTED_ACTION_REPS,
        "actual_action_reps": action_reps,
        "action_reps_ok": action_reps == EXPECTED_ACTION_REPS,
        "relative_stats_check": rel_stats_check,
        "tasks_map": {str(k): v for k, v in sorted(tasks_map.items())},
        "parquet_semantics": "absolute action[t]=state[t+1] on action keys (20D slice)",
        "info_shapes": {"state": state_shape, "action": action_shape},
    }

    per_episode = []
    for loader_index in loader_indices:
        traj = loader[loader_index]
        ep_meta = loader.episodes_metadata[loader_index]
        step = extract_step_data(
            traj,
            step_index,
            modality_configs,
            EmbodimentTag.NEW_EMBODIMENT,
            allow_padding=True,
        )
        state_keys = modality_configs["state"].modality_keys
        action_keys = modality_configs["action"].modality_keys
        semantics = check_action_semantics(traj, state_keys, action_keys)
        history = check_state_history_shapes(step.states)
        rel_conv = check_relative_processor_conversion(
            raw_state=step.states,
            raw_action=step.actions,
            step_index=step_index,
        )
        per_episode.append(
            {
                "loader_index": loader_index,
                "episode_index": int(ep_meta["episode_index"]),
                "tasks": ep_meta.get("tasks"),
                "language": step.text,
                "trajectory_length": len(traj),
                "action_semantics": semantics,
                "state_history_shapes": history,
                "relative_conversion": rel_conv,
            }
        )

    report["episodes_checked"] = per_episode

    sharded = ShardedSingleStepDataset(
        dataset_path=str(dataset_path),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=modality_configs,
        shard_size=256,
        episode_sampling_rate=1.0,
        seed=42,
    )
    report["num_shards"] = len(sharded)

    report["ok"] = (
        delta_ok
        and report["action_reps_ok"]
        and rel_stats_check["ok"]
        and all(ep["action_semantics"]["ok"] for ep in per_episode)
        and all(ep["state_history_shapes"]["ok"] for ep in per_episode)
        and all(ep["relative_conversion"]["ok"] for ep in per_episode)
        and report["num_shards"] > 0
        and state_shape == [STATE_DIM]
        and action_shape == [ACTION_DIM]
        and len(modality_configs["state"].modality_keys) == 6
        and len(modality_configs["action"].modality_keys) == 4
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline7 relative loader smoke.")
    parser.add_argument("--dataset-path", type=Path, default=MULTI_DATASET)
    parser.add_argument("--step-index", type=int, default=50)
    parser.add_argument("--loader-indices", type=str, default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=SMOKE_ROOT / "multi_345_loader_report.json",
    )
    args = parser.parse_args()

    loader_indices = None
    if args.loader_indices.strip():
        loader_indices = tuple(int(x) for x in args.loader_indices.split(",") if x.strip())

    report = run_loader_smoke(
        args.dataset_path,
        loader_indices=loader_indices,
        step_index=args.step_index,
    )
    write_json(args.output, report)

    print(f"Step6 relative loader smoke: ok={report['ok']}")
    print(f"  delta_indices: {report['observation_delta_indices']}")
    print(f"  shapes: state={report['info_shapes']['state']} action={report['info_shapes']['action']}")
    print(f"  action_reps: {report['actual_action_reps']}")
    print(f"  relative_stats: {report['relative_stats_check']}")
    for ep in report["episodes_checked"]:
        print(
            f"  ep={ep['episode_index']} parquet_max_diff={ep['action_semantics']['max_abs_diff']:.2e} "
            f"history_ok={ep['state_history_shapes']['ok']} rel_conv={ep['relative_conversion']['ok']}"
        )
    print(f"  report: {args.output}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
