"""Validate stage artifacts without decompressing complete volumes."""

from pathlib import Path

import numpy as np
import tifffile

from utils.misc import VolumeIO, normalize_resolution


def validate_sdf_outputs(directory, image_directory, resolution, downsample):
    """Require one float16 distance volume per input ID on the declared grid."""
    images = VolumeIO.list_volume_files(image_directory)
    files = VolumeIO.list_volume_files(directory)
    expected_ids = {Path(path).stem for path in images}
    actual_ids = {Path(path).stem for path in files}
    if not images or expected_ids != actual_ids or len(files) != len(images):
        raise ValueError(f"Incomplete or ambiguous SDF output set in {directory}")
    shape = tuple(size // downsample for size in normalize_resolution(resolution))
    for path in files:
        if Path(path).suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"New stage outputs must be lossless TIF: {path}")
        with tifffile.TiffFile(path) as volume:
            series = volume.series[0]
            if series.shape != shape or series.dtype != np.float16:
                raise ValueError(
                    f"Invalid SDF artifact {path}: {series.shape}/{series.dtype}; "
                    f"expected {shape}/float16"
                )
