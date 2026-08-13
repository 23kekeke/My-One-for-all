"""Offline GR00T bootstrap hooks for pipeline2 (no scripts/ imports)."""

from __future__ import annotations

import os
from pathlib import Path


_COSMOS_PATCH_VERSION = "pipeline2_local_cosmos_v1"


_COSMOS_HUB_IDS = (
    "nvidia/Cosmos-Reason2-2B",
    "nvidia/Cosmos-Reason2",
)


def _resolved_cosmos_local_path() -> Path | None:
    configured_path = os.environ.get("GR00T_COSMOS_MODEL_PATH")
    if not configured_path:
        return None
    local_path = Path(configured_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"GR00T_COSMOS_MODEL_PATH does not exist: {local_path}")
    return local_path


def install_local_cosmos_hub_redirect() -> None:
    """Redirect HF hub id nvidia/Cosmos-Reason2-2B → GR00T_COSMOS_MODEL_PATH.

    Processor redirect alone is not enough: Gr00tN1d7 backbone
    ``from_pretrained(config.model_name)`` still uses the hub id and fails
    offline. Wrap gr00t's local-first loader so model/config loads also hit
    the deploy-packaged Cosmos directory.
    """
    local_path = _resolved_cosmos_local_path()
    if local_path is None:
        return

    import gr00t as gr00t_pkg

    current = getattr(gr00t_pkg, "_hf_local_first_call", None)
    if current is None:
        return
    if getattr(current, "_pipeline2_local_cosmos_hub_redirect", False):
        return

    def patched_hf_local_first_call(
        orig_func,
        klass,
        pretrained_model_name_or_path,
        *args,
        skip_model_weights: bool = False,
        **kwargs,
    ):
        name_str = str(pretrained_model_name_or_path)
        if name_str in _COSMOS_HUB_IDS:
            print(
                "[pipeline2_local_cosmos] "
                f"hub redirect: {name_str!r} -> {str(local_path)!r}",
                flush=True,
            )
            pretrained_model_name_or_path = str(local_path)
        return current(
            orig_func,
            klass,
            pretrained_model_name_or_path,
            *args,
            skip_model_weights=skip_model_weights,
            **kwargs,
        )

    patched_hf_local_first_call._pipeline2_local_cosmos_hub_redirect = True
    gr00t_pkg._hf_local_first_call = patched_hf_local_first_call


def install_local_cosmos_processor_patch() -> None:
    """Redirect nvidia/Cosmos-Reason2-2B processor loads to GR00T_COSMOS_MODEL_PATH."""
    local_path = _resolved_cosmos_local_path()
    if local_path is None:
        return

    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    current = processing.build_processor
    if getattr(current, "_pipeline2_local_cosmos_patch_version", None) == _COSMOS_PATCH_VERSION:
        return

    def patched_build_processor(model_name, transformers_loading_kwargs):
        print(
            "[pipeline2_local_cosmos] "
            f"processor redirect: {model_name!r} -> {str(local_path)!r}",
            flush=True,
        )
        return current(str(local_path), transformers_loading_kwargs)

    patched_build_processor._pipeline2_local_cosmos_patch_version = _COSMOS_PATCH_VERSION
    processing.build_processor = patched_build_processor


def install_processor_image_override_patch() -> None:
    """Ensure letter_box / resize kwargs passed from training setup are honored."""
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    from gr00t.model.gr00t_n1d7.image_augmentations import (
        build_image_transformations,
        build_image_transformations_albumentations,
    )

    cls = processing.Gr00tN1d7Processor
    if getattr(cls.from_pretrained, "_pipeline2_image_override_patch", False):
        return

    original = cls.from_pretrained.__func__

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        override_keys = (
            "letter_box_transform",
            "shortest_image_edge",
            "crop_fraction",
            "image_crop_size",
            "image_target_size",
            "use_albumentations",
        )
        image_overrides = {key: kwargs.pop(key) for key in override_keys if key in kwargs}
        processor = original(cls, pretrained_model_name_or_path, **kwargs)
        if not image_overrides:
            return processor

        for key, value in image_overrides.items():
            setattr(processor, key, value)

        if processor.use_albumentations:
            processor.train_image_transform, processor.eval_image_transform = (
                build_image_transformations_albumentations(
                    processor.image_target_size,
                    processor.image_crop_size,
                    processor.random_rotation_angle,
                    processor.color_jitter_params,
                    processor.shortest_image_edge,
                    processor.crop_fraction,
                    extra_augmentation_config=processor.extra_augmentation_config,
                    letter_box_transform=processor.letter_box_transform,
                )
            )
        else:
            processor.train_image_transform, processor.eval_image_transform = (
                build_image_transformations(
                    processor.image_target_size,
                    processor.image_crop_size,
                    processor.random_rotation_angle,
                    processor.color_jitter_params,
                    letter_box_transform=processor.letter_box_transform,
                )
            )
        print(
            "[pipeline2_processor_override] "
            f"letter_box={processor.letter_box_transform} "
            f"shortest_edge={processor.shortest_image_edge} "
            f"crop_fraction={processor.crop_fraction}",
            flush=True,
        )
        return processor

    patched_from_pretrained._pipeline2_image_override_patch = True
    cls.from_pretrained = patched_from_pretrained


def install_offline_hooks() -> None:
    """Enable offline-safe HF loading before importing GR00T training/inference code."""
    os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
    os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import gr00t  # noqa: F401 — triggers gr00t.__init__ patches when env is set

    install_local_cosmos_hub_redirect()
    install_local_cosmos_processor_patch()
    install_processor_image_override_patch()
