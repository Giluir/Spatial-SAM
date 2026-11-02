# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

import warnings

import numpy as np
import torch


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Connected components require an N x 1 x H x W mask")
    if mask.is_cuda:
        try:
            from sam2 import _C
        except ImportError:
            _C = None
        if _C is not None:
            return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())

    # Preserve the same 8-connected regions when the optional kernel is absent.
    from scipy.ndimage import label

    source = mask.detach().cpu().numpy().astype(bool)
    labels = np.zeros(source.shape, dtype=np.int32)
    counts = np.zeros(source.shape, dtype=np.int32)
    for index in range(source.shape[0]):
        components, _ = label(source[index, 0], structure=np.ones((3, 3)))
        areas = np.bincount(components.ravel())
        areas[0] = 0
        labels[index, 0] = components
        counts[index, 0] = areas[components]
    return torch.from_numpy(labels).to(mask.device), torch.from_numpy(counts).to(
        mask.device
    )


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def load_volume(
    images,
    image_size,
    offload_video_to_cpu,
    normalization_params=None,
    compute_device=torch.device("cuda"),
):
    # Check if images size match
    if images.shape[1] != image_size or images.shape[2] != image_size:
        raise RuntimeError(
            f"Image size mismatch: expected {image_size}x{image_size}, got {images.shape[1]}x{images.shape[2]}"
        )

    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    elif images.dtype == np.uint16:
        images = images.astype(np.float32) / 65535.0
    else:
        raise RuntimeError(f"Unknown image dtype: {images.dtype}")

    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images)
    images = images.unsqueeze(1)

    if not offload_video_to_cpu:
        images = images.to(compute_device)

    if normalization_params is None:
        raise ValueError("Explicit image mean and std are required")
    images -= normalization_params["mean"]
    images /= normalization_params["std"]
    return images, image_size, image_size


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
