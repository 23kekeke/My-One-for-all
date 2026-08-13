"""Dongguan pipeline5 relative-action inference (wraps quanta_biman_inference)."""

from dongguan_inference.env import ensure_dongguan_infer_imports
from dongguan_inference.patch import apply_infer_patches

__all__ = ["ensure_dongguan_infer_imports", "apply_infer_patches", "bootstrap_infer"]

_PATCHED = False


def bootstrap_infer() -> None:
    """Register pipeline5 modality/hooks and patch pipeline3 infer for [-7,-1,0]."""
    global _PATCHED
    ensure_dongguan_infer_imports()
    apply_infer_patches()
    _PATCHED = True
