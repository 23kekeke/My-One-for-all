"""Offline inference on Dongguan LeRobot episodes (no robot required)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dongguan_inference import bootstrap_infer
from dongguan_inference.constants import DEFAULT_CHECKPOINT, INFERENCE_TMP, MULTI_DATASET
from dongguan_inference.policy import load_policy, resolve_checkpoint

_PIPELINE5 = Path(__file__).resolve().parents[1]
if str(_PIPELINE5) not in sys.path:
    sys.path.insert(0, str(_PIPELINE5))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_modality_configs():
    bootstrap_infer()
    import dongguan_relative_config

    return dongguan_relative_config.dongguan_relative_config


def infer_offline_episode(
    *,
    policy: Any,
    dataset_path: Path | str,
    loader_index: int = 0,
    step_index: int = 0,
    execution_horizon: int = 1,
) -> dict[str, Any]:
    from copy import deepcopy

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t

    from quanta_biman_inference.action_decode import decode_action_at_step, decoded_step_to_dict

    modality_configs = get_modality_configs()
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
    )
    if loader_index >= len(loader):
        raise IndexError(f"loader_index {loader_index} out of range ({len(loader)} episodes)")

    traj = loader[loader_index]
    ep_index = int(loader.episodes_metadata[loader_index]["episode_index"])
    ep_meta = loader.episodes_metadata[loader_index]
    task_text_from_meta = ep_meta.get("tasks", [None])[0]
    task_index = None
    if task_text_from_meta is not None:
        for idx, text in loader.tasks_map.items():
            if text == task_text_from_meta:
                task_index = int(idx)
                break

    infer_configs = deepcopy(modality_configs)
    infer_configs.pop("action")
    data_point = extract_step_data(
        traj, step_index, infer_configs, EmbodimentTag.NEW_EMBODIMENT, allow_padding=True
    )

    obs: dict[str, Any] = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for language_key in modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text
    parsed_obs = parse_observation_gr00t(obs, infer_configs)

    action_dict, _info = policy.get_action(parsed_obs)
    planned = [
        decoded_step_to_dict(decode_action_at_step(action_dict, j))
        for j in range(execution_horizon)
    ]

    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "loader_index": loader_index,
        "episode_index": ep_index,
        "step_index": step_index,
        "task_index": task_index,
        "task_text": str(data_point.text),
        "execution_horizon": execution_horizon,
        "planned_steps": planned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dongguan relative offline infer smoke.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", type=Path, default=MULTI_DATASET)
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_TMP / "offline_infer_smoke.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bootstrap_infer()
    checkpoint = resolve_checkpoint(args.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy = load_policy(checkpoint)
    report = infer_offline_episode(
        policy=policy,
        dataset_path=args.dataset_path,
        loader_index=args.loader_index,
        step_index=args.step_index,
        execution_horizon=args.execution_horizon,
    )
    report["checkpoint"] = str(checkpoint.resolve())
    write_json(args.output, report)
    print(f"Offline infer ok: {args.output}")
    print(f"  episode={report['episode_index']} step={report['step_index']} task_index={report['task_index']}")


if __name__ == "__main__":
    main()
