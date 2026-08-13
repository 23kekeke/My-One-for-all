# Pipeline7 relative train/infer modality bootstrap.
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# code/pipeline7/pipeline7_dongguan_relative_eef_only/this_file.py → code/
CODE = Path(__file__).resolve().parents[2]
PIPELINE2 = CODE / "pipeline2"
PIPELINE7 = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE2))
from gr00t_offline_hooks import install_offline_hooks

install_offline_hooks()

sys.path.insert(0, str(PIPELINE7))
importlib.import_module("dongguan_eef_only_relative_config")
