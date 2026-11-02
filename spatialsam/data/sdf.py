"""Explicit conversions between segmentation logits, masks, and distance fields."""

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, zoom
from torch.nn import functional as F


def restore_reference_image_grid(sdf: np.ndarray, image_shape) -> np.ndarray:
    """Restore exported UNet SDFs before SAM training resamples its reference.

    Export used nearest-neighbor interpolation and preserved float16 values.
    This intermediate grid changes the subsequent linear interpolation, even
    though the cached native SDF and the final SAM grid are both isotropic.
    """
    scale = np.asarray(image_shape) / np.asarray(sdf.shape)
    if np.allclose(scale, 1):
        return sdf
    values = sdf.astype(np.float32) if sdf.dtype == np.float16 else sdf
    return zoom(values, scale, order=0).astype(sdf.dtype, copy=False)


def mask_to_sdf(
    mask: np.ndarray,
    target_shape: Sequence[int] | None = None,
    clip_max: float = 15.0,
) -> np.ndarray:
    """Return the original positive-inside, clipped float16 distance field.

    Thresholding precedes nearest-neighbor resizing. Distances are measured on
    the target grid; they are not resized logits or a resized distance field.
    """
    if mask.ndim != 3:
        raise ValueError("SDF conversion requires a 3D mask")
    binary = mask > 0
    if target_shape is not None:
        scale = np.asarray(target_shape) / np.asarray(binary.shape)
        if not np.allclose(scale, 1):
            binary = zoom(binary, scale, order=0)
    distances = distance_transform_edt(binary) - distance_transform_edt(binary == 0)
    return np.clip(distances, -clip_max, clip_max).astype(np.float16)


def blur_logits(
    logits: np.ndarray, sigma: float = 1.0, *, device="cpu", use_autocast=False
) -> np.ndarray:
    """Apply a finite Gaussian kernel with zero padding and preserved precision."""
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(2 * radius + 1, device=device) - radius
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (
        kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    )
    kernel /= kernel.sum()
    volume = torch.as_tensor(
        np.ascontiguousarray(logits), dtype=torch.float32, device=device
    )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ),
    ):
        result = F.conv3d(volume[None, None], kernel[None, None], padding=radius)[0, 0]
    return result.float().cpu().numpy()
