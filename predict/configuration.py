"""Validate public inference and bootstrap options before loading any models."""

import math

from utils.misc import normalize_resolution
from spatialsam.configuration import validate_output_directory


COMMON_FIELDS = {
    "volume_dir",
    "output_root",
    "resolution",
    "unet_downsample",
    "mean",
    "std",
    "save_masks",
    "strict_stem_match",
    "device",
    "root",
    "limit_volumes",
    "dataset_name",
    "split",
    "sam2_config_path",
    "sam2_model_path",
    "to_sdf",
    "mask_to_binary",
}
INFERENCE_FIELDS = COMMON_FIELDS | {
    "mode",
    "sdf_model_path",
    "sdf_infer",
    "sdf_model",
    "ref_masks_dir",
    "restore_output_resolution",
    "to_instance",
    "postprocess",
}
BOOTSTRAP_FIELDS = COMMON_FIELDS | {
    "true_dir",
    "annotations",
    "label_start",
    "label_stride",
    "select_mask_with_obj",
    "slices",
    "save_few_masks",
    "save_xyz_masks",
    "twice",
    "pseudo_blur_sigma",
}


def validate_options(config, *, bootstrap=False):
    """Reject ignored fields, invalid grids, and incomplete model dependencies."""
    unknown = set(config) - (BOOTSTRAP_FIELDS if bootstrap else INFERENCE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown prediction options: {sorted(unknown)}")
    for key in ("volume_dir", "output_root"):
        if not config.get(key):
            raise ValueError(f"Prediction requires {key}")
    validate_output_directory(
        config["output_root"],
        config["volume_dir"],
        config.get("true_dir"),
        config.get("ref_masks_dir"),
    )
    shape = normalize_resolution(config.get("resolution", 1024))
    downsample = int(config.get("unet_downsample", 2))
    if downsample <= 0 or any(size % downsample for size in shape):
        raise ValueError(
            "Resolution must be divisible by a positive SDF downsample factor"
        )
    mean, std = config.get("mean"), config.get("std")
    if (mean is None) != (std is None):
        raise ValueError("Normalization requires both mean and std")
    if mean is not None and (
        not math.isfinite(mean) or not math.isfinite(std) or std <= 0
    ):
        raise ValueError(
            "Normalization statistics must be finite and std must be positive"
        )
    mode = config.get(
        "mode", "sam_with_cached_sdf" if config.get("ref_masks_dir") else "end2end"
    )
    if bootstrap:
        if bool(config.get("true_dir")) == bool(config.get("annotations")):
            raise ValueError(
                "Bootstrap requires exactly one of true_dir or annotations"
            )
        if not config.get("annotations") and (
            config.get("slices", 5) <= 0 or config.get("label_stride", 200) <= 0
        ):
            raise ValueError("Prompt slice budget and stride must be positive")
    elif mode not in ("sdf_only", "end2end", "sam_with_cached_sdf"):
        raise ValueError(f"Unsupported inference mode: {mode}")
    if bootstrap or mode != "sdf_only":
        for key in ("sam2_config_path", "sam2_model_path"):
            if not config.get(key):
                raise ValueError(f"SAM prediction requires {key}")
    if not bootstrap:
        final_masks = mode != "sdf_only" and not config.get("to_sdf", False)
        if config.get("to_instance", False) and (
            not final_masks or not config.get("postprocess", True)
        ):
            raise ValueError("to_instance requires final SAM masks with postprocessing")
        if final_masks and config.get("postprocess", True) and downsample != 2:
            raise ValueError(
                "The validated mask postprocessing requires unet_downsample=2"
            )
        if mode == "sam_with_cached_sdf" and not config.get("ref_masks_dir"):
            raise ValueError("Cached SAM inference requires ref_masks_dir")
        if mode in ("sdf_only", "end2end") and not config.get("sdf_model_path"):
            raise ValueError("SDF prediction requires sdf_model_path")
        window = config.get("sdf_infer", {}) or {}
        unknown = set(window) - {
            "patch_shape",
            "stride_shape",
            "halo_shape",
            "output_dtype",
            "use_autocast",
            "autocast_dtype",
        }
        if unknown:
            raise ValueError(f"Unknown SDF window options: {sorted(unknown)}")
        for name in ("patch_shape", "stride_shape", "halo_shape"):
            if name in window:
                values = window[name]
                minimum = 0 if name == "halo_shape" else 1
                if len(values) != 3 or any(value < minimum for value in values):
                    raise ValueError(f"Invalid SDF {name}: {values}")
        if window.get("output_dtype", "float32") not in ("float16", "float32"):
            raise ValueError("SDF output_dtype must be float16 or float32")
        if window.get("autocast_dtype", "bfloat16") not in ("bfloat16", "float16"):
            raise ValueError("SDF autocast_dtype must be bfloat16 or float16")
