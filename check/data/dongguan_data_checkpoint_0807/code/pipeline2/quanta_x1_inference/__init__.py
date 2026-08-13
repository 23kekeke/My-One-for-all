"""Quanta X1 16D GR00T inference (train-aligned letterbox, state/action, QLoRA).

Package ``__init__`` stays import-light so ``xr_lerobot`` can run
``live_capture`` / ``live_sdk_daemon`` without installing GR00T.
Import submodules explicitly, e.g. ``from quanta_x1_inference.policy import load_policy``.
"""
