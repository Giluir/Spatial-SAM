# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import numpy as np

import torch

from scipy.ndimage import zoom


from training.dataset.masks import (
    VolumeSegmentLoader,
    RefMaskLoader,
)
from utils.misc import (
    compute_zoom_scale,
    is_uniform_scale,
    normalize_resolution,
)
from spatialsam.data.io import build_training_dataset_entries
from spatialsam.data.sdf import restore_reference_image_grid


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: Optional[str] = None
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_data_path(self, idx):
        raise NotImplementedError()

    def get_video(self, idx):
        raise NotImplementedError()


class VolumeDataset(VOSRawDataset):
    def __init__(
        self,
        dataset_config,
        volume_io,
        sample_rate=1,
        single_object_mode=True,
        resolution=1024,
        split="train",
        use_ref_mask=False,
        use_few_masks=False,
        mask_to_binary=True,
        scale_mask_value=False,
        restore_ref_to_image_grid=False,
    ):
        self.dataset_config = dataset_config
        self.volume_loader = volume_io
        self.sample_rate = sample_rate
        self.single_object_mode = single_object_mode
        self.use_ref_mask = use_ref_mask
        self.use_few_masks = use_few_masks
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split={split}, expected 'train' or 'val'")
        self.datasets = self.from_config(
            dataset_config,
            use_ref_mask=use_ref_mask,
            use_few_masks=use_few_masks,
            split=split,
        )
        self.active_volume = None
        self.active_segment_loader = None
        self.active_ref_mask_loader = None
        self.active_few_masks_info = None
        self.resolution = normalize_resolution(resolution, dims=3)
        self.mask_to_binary = mask_to_binary
        self.scale_mask_value = scale_mask_value
        self.restore_ref_to_image_grid = restore_ref_to_image_grid

    @staticmethod
    def from_config(
        dataset_config, use_ref_mask=False, use_few_masks=False, split="train"
    ):
        return build_training_dataset_entries(
            dataset_config,
            split=split,
            use_ref_mask=use_ref_mask,
            use_few_masks=use_few_masks,
        )

    def load_frames(self, volume, mean, std):
        import numpy as np

        if volume.dtype != np.uint8 and volume.dtype != np.uint16:
            raise ValueError("Data type not supported.")

        if volume.dtype == np.uint8:
            volume_tensor = torch.from_numpy(volume).float() / 255.0
        elif volume.dtype == np.uint16:
            volume_tensor = torch.from_numpy(volume).float() / 65535.0
        volume_tensor = (volume_tensor - mean) / std

        frames = []
        for frame_idx in range(volume_tensor.shape[0]):
            frame = volume_tensor[frame_idx].unsqueeze(0)
            frames.append(VOSFrame(frame_idx, data=frame))

        return frames

    def replace_masks_with_few_masks(self, masks, few_masks_paths):
        """
        Replace masks frames with few_masks for improved training.

        Replaces specific depth slices in the masks volume with higher quality
        few-shot prompts from pseudo label initialization.

        Args:
            masks: Original masks volume shape (D, H, W)
            few_masks_paths: List of few_mask file paths
                           e.g., ["volume_001_z_frame_100.tif", "volume_001_z_frame_200.tif"]

        Returns:
            List of replaced frame indices, e.g., [100, 200]
        """
        if few_masks_paths is None or len(few_masks_paths) == 0:
            return []

        expected_shape = (self.resolution[1], self.resolution[2])
        replaced_frames = []

        for file_path in few_masks_paths:
            filename = os.path.basename(file_path)
            try:
                frame_idx = int(filename.split("_frame_")[1].split(".")[0])
                few_mask = self.volume_loader.load(file_path)

                if tuple(few_mask.shape) != expected_shape:
                    raise ValueError(
                        f"Few mask {filename}: resolution {few_mask.shape} != expected {expected_shape}"
                    )

                if frame_idx < masks.shape[0]:
                    masks[frame_idx, :, :] = few_mask
                    replaced_frames.append(frame_idx)
                else:
                    logging.warning(
                        "Frame index %d exceeds mask depth %d",
                        frame_idx,
                        masks.shape[0],
                    )

            except (ValueError, IndexError) as e:
                logging.warning("Could not process slice mask %s: %s", filename, e)
                continue

        return sorted(replaced_frames)

    def get_data_path(self, idx):
        return self.datasets[idx].volume_path, self.datasets[idx].masks_path

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        if self.active_volume is None or self.active_volume.video_id != idx:
            dataset = self.datasets[idx]
            volume_path = dataset.volume_path
            masks_path = dataset.masks_path
            ref_masks_path = dataset.ref_masks_path if self.use_ref_mask else None
            few_masks_paths = dataset.few_masks_paths if self.use_few_masks else None
            volume_mean = dataset.volume_mean
            volume_std = dataset.volume_std

            volume = self.volume_loader.load(volume_path)
            masks = self.volume_loader.load(masks_path)
            if ref_masks_path is not None:
                ref_masks = self.volume_loader.load(ref_masks_path)
                if self.restore_ref_to_image_grid:
                    ref_masks = restore_reference_image_grid(ref_masks, volume.shape)

            # Scale the volume and masks to self.resolution
            logging.debug("Volume shape: %s; mask shape: %s", volume.shape, masks.shape)
            volume = self.resize_volume(volume, order_hint=True)
            masks = self.resize_volume(
                masks, order_hint=self.mask_to_binary, scale_value=self.scale_mask_value
            )
            if ref_masks_path is not None:
                ref_masks = self.resize_volume(
                    ref_masks,
                    order_hint=self.mask_to_binary,
                    scale_value=self.scale_mask_value,
                )

            # Replace masks with few_masks if available
            few_masks_info = None
            if few_masks_paths is not None:
                few_masks_info = self.replace_masks_with_few_masks(
                    masks, few_masks_paths
                )
                logging.debug("Annotated frames: %s", few_masks_info)

            frames = self.load_frames(volume, volume_mean, volume_std)
            segment_loader = VolumeSegmentLoader(
                masks, self.single_object_mode, self.mask_to_binary
            )
            if self.use_ref_mask:
                ref_mask_loader = RefMaskLoader(ref_masks, to_logits=False)
            else:
                ref_mask_loader = None

            self.active_volume = VOSVideo(idx, idx, frames)
            self.active_segment_loader = segment_loader
            self.active_ref_mask_loader = ref_mask_loader
            self.active_few_masks_info = few_masks_info

        return (
            self.active_volume,
            self.active_segment_loader,
            self.active_ref_mask_loader,
            self.active_few_masks_info,
        )

    def resize_volume(self, volume: np.ndarray, order_hint, scale_value=False):
        """
        order_hint: if True, use nearest neighbor interpolation (order=0)
        if False, use linear interpolation (order=1) and multiply volume by scale
        """
        scale = compute_zoom_scale(volume.shape, self.resolution)
        if scale_value and not is_uniform_scale(volume.shape, self.resolution):
            raise ValueError(
                "scale_value requires a uniform zoom factor across all axes"
            )

        is_float16 = volume.dtype == np.float16
        if is_float16:
            volume = volume.astype(np.float32)

        if order_hint:
            order = 0
        else:
            order = 1

        if scale_value:
            volume *= scale[0]

        # Check if scaling is needed
        needs_scaling = any(np.abs(s - 1) > 1e-3 for s in scale)

        if needs_scaling:
            volume = zoom(volume, scale, order=order)
        return volume

    def __len__(self):
        return len(self.datasets)
