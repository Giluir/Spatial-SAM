"""Native 3D patch sampling for SDF training and internal validation."""

from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from spatialsam.data.io import build_training_dataset_entries
from training.dataset.sdf_transforms import SDFTransform
from utils.misc import VolumeIO, normalize_resolution


def patch_starts(shape: Sequence[int], patch: Sequence[int], stride: Sequence[int]):
    """Include the final boundary patch on each axis, as in FilterSliceBuilder."""
    axes = []
    for length, width, step in zip(shape, patch, stride):
        if width <= 0 or step <= 0 or length < width:
            raise ValueError(
                f"Invalid patch grid: shape={shape}, patch={patch}, stride={stride}"
            )
        positions = list(range(0, length - width + 1, step))
        if positions[-1] != length - width:
            positions.append(length - width)
        axes.append(positions)
    return [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]


def resize_sdf_volume(volume, resolution, *, is_sdf=False):
    """Resize a continuous volume, scaling SDF distances only for uniform zoom."""
    if volume.dtype == np.float16:
        volume = volume.astype(np.float32)
    if resolution is None:
        return volume
    scale = np.asarray(resolution) / np.asarray(volume.shape)
    if np.allclose(scale, 1):
        return volume
    if is_sdf:
        if not np.all(scale == scale[0]):
            raise ValueError(
                "SDF value scaling requires an equal zoom factor on all axes"
            )
        volume = volume * scale[0]
    return zoom(volume, scale, order=1)


class SDFPatchDataset(Dataset):
    """Sample image/SDF patches with the original filtering and augmentation.

    Normalization follows intensity augmentation. Image conversion to floating
    point precedes resizing, which prevents interpolation from rounding to uint8.
    """

    def __init__(
        self,
        dataset_config,
        patch_shape=(128, 128, 128),
        stride_shape=(80, 80, 80),
        resolution=512,
        split="train",
        min_nonzero_ratio=0.6,
        ignore_index=None,
        slack_acceptance=0.01,
        filter_seed=47,
        transform_seed=47,
        augmentation=None,
        epoch_num_samples=None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported SDF dataset split: {split}")
        if not 0 <= min_nonzero_ratio <= 1 or not 0 <= slack_acceptance <= 1:
            raise ValueError("Patch filtering probabilities must be in [0, 1]")
        self.entries = build_training_dataset_entries(dataset_config, split=split)
        if not self.entries:
            raise ValueError(f"No SDF training samples found for {split}")
        self.patch_shape = tuple(patch_shape)
        self.stride_shape = tuple(stride_shape)
        if len(self.patch_shape) != 3 or len(self.stride_shape) != 3:
            raise ValueError("Patch and stride shapes must each have three dimensions")
        resolution = normalize_resolution(resolution)
        self.volumes = []
        self.transforms = []
        self.indices = []
        seeds = np.random.RandomState(transform_seed)
        volume_io = VolumeIO()
        for volume_index, entry in enumerate(self.entries):
            image = volume_io.load(entry.volume_path)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            target = volume_io.load(entry.masks_path)
            image = resize_sdf_volume(image, resolution)
            target = resize_sdf_volume(target, resolution, is_sdf=True)
            if image.ndim != 3 or image.shape != target.shape:
                raise ValueError(
                    f"Image/SDF shape mismatch: {image.shape}, {target.shape}"
                )
            self.volumes.append((image, target))
            self.transforms.append(
                SDFTransform(
                    mean=entry.volume_mean,
                    std=entry.volume_std,
                    seed=int(seeds.randint(10000000)),
                    training=split == "train",
                    **dict(augmentation or {}),
                )
            )
            rng = np.random.RandomState(filter_seed)
            starts = patch_starts(image.shape, self.patch_shape, self.stride_shape)
            accepted = []
            for start in starts:
                patch = target[self._slices(start)]
                if ignore_index is not None:
                    patch = np.where(patch == ignore_index, 0, patch)
                ratio = np.count_nonzero(patch != 0) / patch.size
                if ratio > min_nonzero_ratio or rng.rand() < slack_acceptance:
                    accepted.append((volume_index, start))
            if not accepted:
                raise ValueError(f"No patches passed filtering for {entry.volume_path}")
            self.indices.extend(accepted)
        if epoch_num_samples is not None:
            if not 0 < epoch_num_samples <= len(self.indices):
                raise ValueError(
                    "epoch_num_samples must be positive and no greater than the patch count"
                )
            self.indices = self.indices[:epoch_num_samples]
        self.repeat_factors = torch.ones(len(self.indices))

    def _slices(self, start):
        return tuple(
            slice(position, position + width)
            for position, width in zip(start, self.patch_shape)
        )

    def set_epoch(self, epoch):
        """Keep per-transform random streams advancing across epochs."""

    def __len__(self):
        return len(self.indices)

    def get_checkpoint_state(self):
        """Serialize independent augmentation streams for every source volume."""
        return [transform.state_dict() for transform in self.transforms]

    def load_checkpoint_state(self, states):
        """Restore augmentation only when the source volume count still matches."""
        if len(states) != len(self.transforms):
            raise ValueError("Checkpoint volume count differs from the current dataset")
        for transform, state in zip(self.transforms, states):
            transform.load_state_dict(state)

    def __getitem__(self, index):
        volume_index, start = self.indices[index]
        image, target = self.volumes[volume_index]
        slices = self._slices(start)
        return self.transforms[volume_index](image[slices], target[slices])
