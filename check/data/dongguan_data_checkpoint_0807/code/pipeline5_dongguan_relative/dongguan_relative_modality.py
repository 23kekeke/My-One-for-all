# Pipeline5 relative train/infer modality bootstrap.
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE2 = ROOT / "pipeline2"
PIPELINE5 = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE2))
from gr00t_offline_hooks import install_offline_hooks

install_offline_hooks()

sys.path.insert(0, str(PIPELINE5))
importlib.import_module("dongguan_relative_config")
