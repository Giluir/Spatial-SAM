# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

import logging
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.volumes import VOSRawDataset
from training.dataset.samplers import VOSSampler

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class SAMVolumeDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        samples_per_volume: int,
        always_target=True,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        logging.info("Loaded %d training volumes", len(self.video_dataset))

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

        self.curr_volume = 0
        self.samples_per_volume = samples_per_volume
        self.volume_num = len(self.video_dataset)

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a video
                video, segment_loader, ref_mask_loader, few_masks_info = (
                    self.video_dataset.get_video(idx)
                )
                # sample frames and object indices to be used in a datapoint
                sampled_frms_and_objs = self.sampler.sample(
                    video,
                    segment_loader,
                    epoch=self.curr_epoch,
                    few_masks_info=few_masks_info,
                )
                break  # Succesfully loaded video
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e

        datapoint = self.construct(
            video, sampled_frms_and_objs, segment_loader, ref_mask_loader
        )
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader, ref_mask_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        # If the input is a volume, we don't use PIL
        volume_frames = load_images(sampled_frames, to_pil=False)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = (
                volume_frames[frame_idx].shape[1],
                volume_frames[frame_idx].shape[2],
            )
            images.append(
                Frame(
                    data=volume_frames[frame_idx],
                    objects=[],
                )
            )
            # We load the gt segments associated with the current frame
            segments = segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.float)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.float)

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )

        # Load reference masks
        frame_ids = [frame.frame_idx for frame in sampled_frames]
        if ref_mask_loader is not None:
            ref_masks = ref_mask_loader.load(frame_ids)
        else:
            ref_masks = None

        return VideoDatapoint(
            frames=images,
            ref_masks=ref_masks,
            video_id=video.video_id,
            size=(h, w),
        )

    def set_epoch(self, epoch):
        """
        Set the current epoch for the sampler.
        """
        self.curr_epoch = epoch
        self.curr_volume = epoch % self.volume_num
        volume_path, masks_path = self.video_dataset.get_data_path(self.curr_volume)
        logging.info(f"Loading volume from '{volume_path}', masks from '{masks_path}'")

    def __getitem__(self, idx):
        if self.training:
            assert idx < self.samples_per_volume
            return self._get_datapoint(self.curr_volume)
        else:
            curr_volume = idx // self.samples_per_volume
            return self._get_datapoint(curr_volume)

    def __len__(self):
        if self.training:
            return self.samples_per_volume
        return len(self.video_dataset) * self.samples_per_volume

    def get_curr_mask_path(self):
        _, masks_path = self.video_dataset.get_data_path(self.curr_volume)
        return masks_path


def load_images(frames, to_pil=True):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            if to_pil:
                all_images.append(tensor_2_PIL(frame.data))
            else:
                all_images.append(frame.data)

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
