"""Patch pipeline3 infer modules for Dongguan eef-only [-7,-1,0] relative contract."""

from __future__ import annotations

import sys
import types
from typing import Any, Sequence

import numpy as np

from dongguan_eef_inference import constants as dc


def _patch_action_decode_for_eef_only() -> None:
    """Policy actions are 20D (no joints); fill joint slots with zeros for DecodedBimanAction."""
    import quanta_biman_inference.action_decode as ad
    from quanta_biman_inference.observation import components_to_arm16

    if getattr(ad.decode_action_at_step, "_pipeline7_eef_only_patch", False):
        return

    def decode_action_at_step(
        action_dict: dict[str, np.ndarray],
        step_index: int,
        *,
        action_keys_list: Sequence[str] | None = None,
        batch_index: int = 0,
    ):
        keys = list(action_keys_list) if action_keys_list is not None else list(dc.ACTION_KEYS)
        unbatched = ad.unbatch_policy_action(
            action_dict,
            action_keys_list=keys,
            batch_index=batch_index,
        )
        horizon = next(iter(unbatched.values())).shape[0]
        if step_index < 0 or step_index >= horizon:
            raise IndexError(f"step_index {step_index} out of range for horizon {horizon}")

        left_joints = unbatched.get("left_joint_position")
        right_joints = unbatched.get("right_joint_position")
        if left_joints is None:
            left_joints = np.zeros((horizon, 6), dtype=np.float32)
        if right_joints is None:
            right_joints = np.zeros((horizon, 6), dtype=np.float32)

        left16 = components_to_arm16(
            eef_9d=unbatched["left_eef_9d"][step_index],
            gripper_position=unbatched["left_gripper_position"][step_index],
            joint_position=left_joints[step_index],
        )
        right16 = components_to_arm16(
            eef_9d=unbatched["right_eef_9d"][step_index],
            gripper_position=unbatched["right_gripper_position"][step_index],
            joint_position=right_joints[step_index],
        )
        return ad.DecodedBimanAction(
            left=ad._arm16_to_decoded("left", left16),
            right=ad._arm16_to_decoded("right", right16),
            vector32=np.concatenate([left16, right16], axis=0),
        )

    decode_action_at_step._pipeline7_eef_only_patch = True  # type: ignore[attr-defined]
    ad.decode_action_at_step = decode_action_at_step

    _orig_keys = ad.action_keys

    def action_keys(modality_configs: dict[str, Any] | None = None) -> list[str]:
        if modality_configs is None:
            return list(dc.ACTION_KEYS)
        return _orig_keys(modality_configs)

    ad.action_keys = action_keys


def apply_infer_patches() -> None:
    import dongguan_eef_only_relative_config as drc

    shim = types.ModuleType("quanta_biman_config")
    shim.OBSERVATION_DELTA_INDICES = list(drc.OBSERVATION_DELTA_INDICES)
    shim.STATE_HISTORY_LENGTH = drc.STATE_HISTORY_LENGTH
    shim.quanta_biman_config = drc.dongguan_eef_only_relative_config
    sys.modules["quanta_biman_config"] = shim

    for mod_name in list(sys.modules):
        if mod_name == "quanta_biman_inference" or mod_name.startswith("quanta_biman_inference."):
            del sys.modules[mod_name]

    import quanta_biman_inference.constants as bc
    import quanta_biman_inference.env as env_mod
    import quanta_biman_inference.policy as policy_mod

    from dongguan_eef_inference.env import ensure_dongguan_infer_imports
    from dongguan_eef_inference.policy import load_policy

    bc.OBSERVATION_DELTA_INDICES = list(drc.OBSERVATION_DELTA_INDICES)
    bc.STATE_HISTORY_LENGTH = drc.STATE_HISTORY_LENGTH
    bc.TEMPORAL_WARMUP_CYCLES = dc.TEMPORAL_WARMUP_CYCLES
    bc.TEMPORAL_SPAN_SEC = dc.TEMPORAL_SPAN_SEC
    bc.CANONICAL_TASKS = dict(dc.CANONICAL_TASKS)
    bc.AUTO_EXECUTE_ARMS_BY_TASK_INDEX = dict(dc.AUTO_EXECUTE_ARMS_BY_TASK_INDEX)
    bc.DEFAULT_CHECKPOINT = dc.DEFAULT_CHECKPOINT
    bc.DEFAULT_TRAIN_OUTPUT = dc.DEFAULT_TRAIN_OUTPUT
    bc.LIVE_RUNS_TMP = dc.LIVE_RUNS_TMP
    bc.INFERENCE_TMP = dc.INFERENCE_TMP
    bc.ACTION_KEYS = list(dc.ACTION_KEYS)
    if hasattr(bc, "STATE_DIM"):
        bc.STATE_DIM = dc.STATE_DIM

    env_mod.ensure_gr00t_imports = ensure_dongguan_infer_imports
    policy_mod.ensure_gr00t_imports = ensure_dongguan_infer_imports
    policy_mod.load_policy = load_policy

    import quanta_biman_inference.observation as obs_mod

    obs_mod.OBSERVATION_DELTA_INDICES = list(drc.OBSERVATION_DELTA_INDICES)
    obs_mod.STATE_HISTORY_LENGTH = drc.STATE_HISTORY_LENGTH

    _patch_action_decode_for_eef_only()

    from dongguan_eef_inference.end_pose_bias import install_live_end_pose_waypoint_patch

    install_live_end_pose_waypoint_patch()
