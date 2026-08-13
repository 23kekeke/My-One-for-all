"""GR00T import/env setup for biman inference."""

from __future__ import annotations

import os
import sys

from quanta_biman_inference.constants import COSMOS_MODEL, GR00T_REPO, PIPELINE2, PIPELINE3


def setup_gr00t_env() -> dict[str, str]:
    env = os.environ.copy()
    pipeline_paths = [str(GR00T_REPO), str(PIPELINE2), str(PIPELINE3)]
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


def ensure_gr00t_imports() -> None:
    env = setup_gr00t_env()
    for key, value in env.items():
        os.environ[key] = value

    sys.path = [
        p
        for p in sys.path
        if p not in ("", str(GR00T_REPO), str(PIPELINE2), str(PIPELINE3))
        and "Isaac-GR00T" not in p
    ]
    sys.path.insert(0, str(PIPELINE3))
    sys.path.insert(0, str(PIPELINE2))
    sys.path.insert(0, str(GR00T_REPO))

    for mod_name in list(sys.modules):
        if (
            mod_name == "gr00t"
            or mod_name.startswith("gr00t.")
            or mod_name
            in {
                "quanta_biman_config",
                "quanta_biman_smoke_modality",
                "gr00t_offline_hooks",
                "gr00t_qlora_hooks",
            }
        ):
            del sys.modules[mod_name]

    from gr00t_offline_hooks import install_offline_hooks
    from gr00t_qlora_hooks import install_qlora_hooks

    from gr00t_biman_hooks import install_biman_qlora_infer_hooks

    install_offline_hooks()
    if os.environ.get("PIPELINE2_QLORA", "0") == "1":
        install_qlora_hooks()
    install_biman_qlora_infer_hooks()

    import quanta_biman_config  # noqa: F401
