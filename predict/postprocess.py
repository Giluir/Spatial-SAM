"""Shared Watershed postprocessing for semantic and instance masks."""

DEFAULT_CONFIG = {
    "method": "watershed",
    "grid": "native_sam_grid",
    "downsample": 2,
    "dilation_radius": 2,
    "sdf_threshold": 2,
    "dust_size": 64,
}


def process(scores, ref_sdf, config=None):
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import watershed
    from utils.misc import mask_dilation, remove_dust_from_segmentation

    if config is None:
        config = DEFAULT_CONFIG
    if config["method"] != "watershed" or config["grid"] != "native_sam_grid":
        raise ValueError("Expected Watershed on the native SAM grid")
    if config["downsample"] != 2 or config["dilation_radius"] != 2:
        raise ValueError("Watershed uses stride two and radius-two dilation")
    full_masks = scores > 0
    masks = full_masks[::2, ::2, ::2]
    ref_sdf = np.asarray(ref_sdf)
    if masks.shape != ref_sdf.shape or any(n % 2 for n in full_masks.shape):
        raise ValueError(
            "Native masks and unscaled reference SDF must use matching grids"
        )
    masks = mask_dilation(masks, radius=2)
    markers, _ = ndimage.label(ref_sdf > config["sdf_threshold"])
    labels = watershed(-ref_sdf, markers, mask=masks)
    labels = remove_dust_from_segmentation(labels, min_size=config["dust_size"])
    labels = ndimage.zoom(labels, 2, order=0)
    # The original foreground is the final gate; no independent hole filling.
    labels *= full_masks
    return labels


def final_mask(scores, ref_sdf, *, to_instance=False):
    """Both output modes share exactly the same foreground and processing."""
    labels = process(scores, ref_sdf)
    return labels if to_instance else labels > 0
