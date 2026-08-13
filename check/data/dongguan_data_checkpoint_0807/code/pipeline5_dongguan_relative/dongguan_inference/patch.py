"""Patch pipeline3 infer modules for Dongguan [-7,-1,0] relative contract."""

from __future__ import annotations

import sys
import types

from dongguan_inference import constants as dc


def apply_infer_patches() -> None:
    import dongguan_relative_config as drc

    shim = types.ModuleType("quanta_biman_config")
    shim.OBSERVATION_DELTA_INDICES = list(drc.OBSERVATION_DELTA_INDICES)
    shim.STATE_HISTORY_LENGTH = drc.STATE_HISTORY_LENGTH
    shim.quanta_biman_config = drc.dongguan_relative_config
    sys.modules["quanta_biman_config"] = shim

    for mod_name in list(sys.modules):
        if mod_name == "quanta_biman_inference" or mod_name.startswith("quanta_biman_inference."):
            del sys.modules[mod_name]

    import quanta_biman_inference.constants as bc
    import quanta_biman_inference.env as env_mod
    import quanta_biman_inference.policy as policy_mod

    from dongguan_inference.env import ensure_dongguan_infer_imports
    from dongguan_inference.policy import load_policy

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

    env_mod.ensure_gr00t_imports = ensure_dongguan_infer_imports
    policy_mod.ensure_gr00t_imports = ensure_dongguan_infer_imports
    policy_mod.load_policy = load_policy

    import quanta_biman_inference.observation as obs_mod

    obs_mod.OBSERVATION_DELTA_INDICES = list(drc.OBSERVATION_DELTA_INDICES)
    obs_mod.STATE_HISTORY_LENGTH = drc.STATE_HISTORY_LENGTH

    from dongguan_inference.end_pose_bias import install_live_end_pose_waypoint_patch

    install_live_end_pose_waypoint_patch()
