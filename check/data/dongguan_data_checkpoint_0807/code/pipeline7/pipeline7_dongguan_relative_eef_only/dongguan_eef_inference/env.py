"""GR00T import/env setup for Dongguan pipeline7 eef-only inference."""

from __future__ import annotations

import os
import sys

from dongguan_eef_inference.constants import (
    COSMOS_MODEL,
    GR00T_REPO,
    PIPELINE2,
    PIPELINE3,
    PIPELINE7,
)


def setup_gr00t_env() -> dict[str, str]:
    env = os.environ.copy()
    pipeline_paths = [str(GR00T_REPO), str(PIPELINE7), str(PIPELINE3), str(PIPELINE2)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(pipeline_paths + ([existing] if existing else []))
    env["GR00T_COSMOS_MODEL_PATH"] = str(COSMOS_MODEL)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["LOGURU_LEVEL"] = env.get("LOGURU_LEVEL", "INFO")
    env["GROOT_PATCH_MISTRAL"] = "1"
    env["GROOT_HF_LOCAL_FIRST"] = "1"
    env["PIPELINE2_QLORA"] = "1"
    return env


def ensure_dongguan_infer_imports() -> None:
    env = setup_gr00t_env()
    for key, value in env.items():
        os.environ[key] = value

    sys.path = [
        p
        for p in sys.path
        if p
        not in (
            "",
            str(GR00T_REPO),
            str(PIPELINE7),
            str(PIPELINE3),
            str(PIPELINE2),
        )
        and "Isaac-GR00T" not in p
    ]
    sys.path.insert(0, str(PIPELINE2))
    sys.path.insert(0, str(PIPELINE3))
    sys.path.insert(0, str(PIPELINE7))
    sys.path.insert(0, str(GR00T_REPO))

    stale = {
        "dongguan_eef_only_relative_config",
        "dongguan_eef_only_relative_modality",
        "dongguan_relative_config",
        "dongguan_relative_modality",
        "quanta_biman_config",
        "quanta_biman_relative_config",
        "gr00t_offline_hooks",
        "gr00t_qlora_hooks",
        "gr00t_dongguan_hooks",
        "gr00t_dongguan_infer_hooks",
        "gr00t_biman_hooks",
        "gr00t",
    }
    for mod_name in list(sys.modules):
        if mod_name in stale or mod_name.startswith("gr00t."):
            del sys.modules[mod_name]

    from gr00t_offline_hooks import install_offline_hooks
    from gr00t_qlora_hooks import DEFAULT_LORA_ALPHA, DEFAULT_LORA_R, install_qlora_hooks

    from dongguan_eef_inference.gr00t_dongguan_infer_hooks import install_dongguan_qlora_infer_hooks

    install_offline_hooks()
    if os.environ.get("PIPELINE2_QLORA", "0") == "1":
        install_qlora_hooks(lora_r=DEFAULT_LORA_R, lora_alpha=DEFAULT_LORA_ALPHA)
    install_dongguan_qlora_infer_hooks()

    import dongguan_eef_only_relative_modality  # noqa: F401
