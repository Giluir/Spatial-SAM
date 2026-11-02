from __future__ import annotations

import numpy as np
import os
from pathlib import Path
from skimage.morphology import ball, binary_dilation

from collections.abc import Sequence


class VolumeIO:
    SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".zst", ".npz")

    def __init__(self):
        self.volume_path = None

    @classmethod
    def is_supported_volume_file(cls, path):
        suffix = Path(path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def list_volume_files(cls, directory, limit_volumes=None, extensions=None):
        """List one preferred format per stem, then apply the sample limit."""
        if not directory or not os.path.isdir(directory):
            return []

        allowed = (
            tuple(ext.lower() for ext in extensions)
            if extensions is not None
            else cls.SUPPORTED_EXTENSIONS
        )
        priority = {ext: index for index, ext in enumerate(cls.SUPPORTED_EXTENSIONS)}
        by_stem = {}
        for file_name in sorted(os.listdir(directory)):
            path = Path(directory) / file_name
            suffix = path.suffix.lower()
            if suffix not in allowed or not path.is_file():
                continue
            rank = priority.get(suffix, len(priority))
            previous = by_stem.get(path.stem)
            if previous is None or rank < previous[0]:
                by_stem[path.stem] = (rank, str(path))
        files = sorted(path for _, path in by_stem.values())
        if limit_volumes is not None and int(limit_volumes) > 0:
            files = files[: int(limit_volumes)]
        return files

    def load(self, volume_path):
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"Volume file {volume_path} not found")

        suffix = Path(volume_path).suffix.lower()
        if suffix == ".zst":
            return self.decompress_zstd(volume_path)
        elif suffix in (".tif", ".tiff"):
            return self.load_tif(volume_path)
        elif suffix == ".npz":
            return self.load_npz(volume_path)

        raise ValueError(f"Unsupported volume format: {volume_path}")

    def save(self, volume, output_path):
        suffix = Path(output_path).suffix.lower()
        if suffix == ".zst":
            self.compress_zstd(volume, output_path)
        elif suffix in (".tif", ".tiff"):
            self.save_tif(volume, output_path)
        elif suffix == ".npz":
            self.save_npz(volume, output_path)
        else:
            raise ValueError(f"Unsupported volume format: {output_path}")

    @staticmethod
    def decompress_zstd(input_file):
        import zstandard as zstd
        import pickle

        with open(input_file, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            compressed = f.read()
            array_bytes = dctx.decompress(compressed)
            return pickle.loads(array_bytes)

    @staticmethod
    def compress_zstd(ndarray, output_file):
        import zstandard as zstd
        import pickle

        try:
            threads = os.cpu_count()
        except AttributeError:
            threads = 0
        threads = min(threads, 8)

        with open(output_file, "wb") as f:
            cctx = zstd.ZstdCompressor(threads=threads)
            compressed = cctx.compress(pickle.dumps(ndarray))
            f.write(compressed)

    @staticmethod
    def load_tif(input_file):
        import tifffile as tf

        return tf.imread(input_file)

    @staticmethod
    def load_npz(input_file):
        payload = np.load(input_file)
        if isinstance(payload, np.ndarray):
            return payload
        if hasattr(payload, "files") and payload.files:
            return payload[payload.files[0]]
        raise ValueError(f"No array found in npz file: {input_file}")

    @staticmethod
    def save_tif(volume, output_path):
        import tifffile as tf

        tf.imwrite(
            output_path,
            volume,
            photometric="minisblack",
            bigtiff=volume.nbytes > 2**32 - 2**25,
            compression="deflate",
            compressionargs={"level": 1},
        )

    @staticmethod
    def save_npz(volume, output_path):
        np.savez_compressed(output_path, volume=volume)


def normalize_resolution(resolution, dims: int = 3, name: str = "resolution"):
    if resolution is None:
        return None

    if isinstance(resolution, (int, np.integer)):
        return (int(resolution),) * dims

    if isinstance(resolution, np.ndarray):
        resolution = resolution.tolist()

    if isinstance(resolution, Sequence):
        values = tuple(int(v) for v in resolution)
        if len(values) == dims:
            return values
        if dims == 2 and len(values) == 3:
            return values[1], values[2]

    raise ValueError(f"{name} must be an int or a sequence with length {dims}")


def compute_zoom_scale(source_shape, target_shape):
    source = np.asarray(source_shape, dtype=np.float64)
    target = np.asarray(target_shape, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"Shape rank mismatch: {source_shape} vs {target_shape}")
    return target / source


def is_uniform_scale(source_shape, target_shape, atol: float = 1e-6) -> bool:
    scale = compute_zoom_scale(source_shape, target_shape)
    return bool(np.allclose(scale, scale[0], atol=atol))


def inplane_resolution(resolution):
    resolved = normalize_resolution(resolution, dims=3, name="resolution")
    return resolved[1], resolved[2]


def mask_dilation(mask, radius=2):
    return binary_dilation(mask, ball(radius))


def remove_dust_from_segmentation(
    segmentation: np.ndarray, min_size: int = 50
) -> np.ndarray:
    """
    Remove small components (dust) from the segmentation result.

    Args:
        segmentation (np.ndarray): The input segmentation result.
        min_size (int): The minimum size of components to keep.
    """
    # Get the size of each component (label) in the segmentation
    component_sizes = np.bincount(segmentation.ravel())

    # Create a mask for components that are too small
    too_small_mask = component_sizes < min_size
    labels_to_remove = np.arange(len(component_sizes))[too_small_mask]

    # Set the small components to background (0)
    filtered_segmentation = segmentation.copy()
    small_objects_pixels = np.isin(filtered_segmentation, labels_to_remove)
    filtered_segmentation[small_objects_pixels] = 0

    return filtered_segmentation
