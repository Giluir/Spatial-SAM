"""Complete annotated Z slices, explicitly located on each original image grid."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from utils.misc import VolumeIO, normalize_resolution


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate annotation JSON key: {key}")
        result[key] = value
    return result


def read_annotations(path, image_paths):
    """Validate an exact image-stem mapping; mask paths are relative to the JSON."""
    path = Path(path).expanduser().resolve()
    mapping = json.loads(path.read_text(), object_pairs_hook=_unique_keys)
    if not isinstance(mapping, dict):
        raise ValueError(
            "Annotation JSON must be an object: {volume_stem: [{z, mask}, ...]}"
        )
    expected = {Path(image).stem for image in image_paths}
    if not expected:
        raise ValueError("No image volumes found")
    if set(mapping) != expected:
        raise ValueError(
            f"Each volume requires annotations; missing: {sorted(expected - set(mapping))}; "
            f"unexpected: {sorted(set(mapping) - expected)}"
        )
    resolved = {}
    for name, entries in mapping.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{name}: Provide at least one fully annotated 2D slice")
        seen = set()
        records = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"z", "mask"}:
                raise ValueError(
                    f"{name}: Each annotation must contain exactly z and mask"
                )
            z = entry["z"]
            if type(z) is not int or z < 0 or z in seen:
                raise ValueError(
                    f"{name}: z must be a unique nonnegative, zero-based integer: {z}"
                )
            seen.add(z)
            if not isinstance(entry["mask"], str) or not entry["mask"]:
                raise ValueError(f"{name}: mask must be a file path")
            mask = Path(entry["mask"]).expanduser()
            mask = (
                (path.parent / mask).resolve()
                if not mask.is_absolute()
                else mask.resolve()
            )
            if not mask.is_file():
                raise FileNotFoundError(mask)
            if mask.suffix.lower() not in (*VolumeIO.SUPPORTED_EXTENSIONS, ".png"):
                raise ValueError(f"Unsupported 2D annotation format: {mask}")
            records.append({"z": z, "mask": str(mask)})
        resolved[name] = sorted(records, key=lambda entry: entry["z"])
    return resolved


def image_shape(path):
    """Read TIF dimensions without loading the image volume."""
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as image:
            return image.series[0].shape
    return VolumeIO().load(path).shape


def annotation_frame(z, original_depth, target_depth):
    """Choose a frame that actually samples this original Z plane, without guessing."""
    if not 0 <= z < original_depth:
        raise ValueError(
            f"Annotation z={z} is outside the source depth {original_depth}"
        )
    # Match the exact nearest-neighbor convention used to resize the input image.
    source_indices = zoom(
        np.arange(original_depth), target_depth / original_depth, order=0
    )
    candidates = np.flatnonzero(source_indices == z)
    if not len(candidates):
        raise ValueError(
            f"Source slice z={z} is lost when resampling depth {original_depth} to {target_depth}; "
            "increase the inference Z resolution to retain annotated slices"
        )
    return int(candidates[len(candidates) // 2])


def load_annotation_prompts(entries, original_shape, resolution):
    """Only listed slices become prompts; a listed all-zero mask is real background."""
    if len(original_shape) != 3:
        raise ValueError(f"Input images must be 3D ZYX volumes: {original_shape}")
    resolution = normalize_resolution(resolution)
    prompts = {}
    records = []
    for entry in entries:
        path = Path(entry["mask"])
        if path.suffix.lower() == ".png":
            from PIL import Image

            with Image.open(path) as image:
                mask = np.asarray(image)
        else:
            mask = VolumeIO().load(path)
        if mask.ndim != 2 or mask.shape != tuple(original_shape[1:]):
            raise ValueError(
                f"{path}: expected a complete 2D mask matching the original YX dimensions; "
                f"got {mask.shape}, expected {tuple(original_shape[1:])}"
            )
        if (
            mask.dtype.kind not in "buif"
            or not np.isfinite(mask).all()
            or np.any(mask < 0)
        ):
            raise ValueError(
                f"{path}: mask values must be finite and nonnegative; 0 is background and positive values are foreground"
            )
        frame = annotation_frame(entry["z"], original_shape[0], resolution[0])
        binary = mask > 0
        prompts[frame] = zoom(
            binary, tuple(t / s for t, s in zip(resolution[1:], mask.shape)), order=0
        )
        records.append({**entry, "frame": frame})
    return prompts, records


def validate_annotation_samples(samples, resolution):
    """Preflight all sparse inputs while keeping only one volume's masks in memory."""
    for sample in samples:
        if "annotations" in sample.meta:
            load_annotation_prompts(
                sample.meta["annotations"], image_shape(sample.volume_path), resolution
            )
