# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    metadata: BatchedVideoMetaData
    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@tensorclass
class BatchedVideoDatapointMask(BatchedVideoDatapoint):
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        ref_masks: Optional [TxBxHxW] tensor containing reference masks for each frame in the batch.
                  The first dimension T is the number of frames (same as regular frames),
                  the second dimension B is the batch size (number of videos).
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    masks: torch.BoolTensor
    ref_masks: Optional[torch.BoolTensor] = None

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@tensorclass
class BatchedVideoDatapointSDF(BatchedVideoDatapoint):
    masks: torch.FloatTensor
    ref_masks: Optional[torch.FloatTensor] = None

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """
        return self.img_batch.transpose(0, 1).flatten(0, 1)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """
    Refers to an image/video and all its annotations

    Attributes:
        frames: A list of Frame objects representing the frames in the video.
        video_id: The ID of the video.
        size: The size of the video frames as (height, width).
        ref_masks: Optional list of torch.Tensor objects representing reference masks.
                  Each tensor has shape [H, W] and contains a single mask for a reference frame.
                  The number of ref_masks should match the number of frames.
    """

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]
    ref_masks: Optional[List[torch.Tensor]] = None


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapointMask(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )


def collate_fn_with_ref_mask(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapointSDF:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.

    Returns:
        A BatchedVideoDatapoint instance with ref_masks included.

    Note:
        This function handles ref_masks at the VideoDatapoint level.
        ref_masks is a List[torch.Tensor] where each tensor is a mask.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    # Process regular masks and metadata
    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.float))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )

    # Check if ref_masks are available
    has_ref_masks = any(video.ref_masks is not None for video in batch)

    # Process ref_masks if available
    if has_ref_masks:
        # Ensure all videos have ref_masks
        for video_idx, video in enumerate(batch):
            if video.ref_masks is None:
                raise ValueError(
                    f"Video at index {video_idx} has no ref_masks while others do"
                )

        # Get the number of ref_masks for each video (should match the number of frames)
        for video_idx, video in enumerate(batch):
            if len(video.ref_masks) != len(video.frames):
                raise ValueError(
                    f"Video at index {video_idx} has {len(video.ref_masks)} ref_masks but {len(video.frames)} frames. They should be equal."
                )

        # Ensure all videos have the same number of ref_masks
        ref_masks_counts = [len(video.ref_masks) for video in batch]
        if not all(count == ref_masks_counts[0] for count in ref_masks_counts):
            raise ValueError(
                f"All videos must have the same number of ref_masks. Got: {ref_masks_counts}"
            )

        num_ref_masks = ref_masks_counts[0]
        # num_ref_masks should equal T (number of frames)
        assert (
            num_ref_masks == T
        ), f"Number of ref_masks ({num_ref_masks}) should equal number of frames ({T})"

        # Initialize a tensor to hold all ref_masks
        # Shape: [num_ref_masks, num_videos, H, W]
        # We'll get H and W from the first mask
        first_mask_shape = batch[0].ref_masks[0].shape
        H, W = first_mask_shape

        # Create a tensor of the right shape filled with zeros
        ref_masks_tensor = torch.zeros(
            (num_ref_masks, len(batch), H, W), dtype=torch.float
        )

        # Fill in the tensor with the actual masks
        for video_idx, video in enumerate(batch):
            for mask_idx, mask in enumerate(video.ref_masks):
                # Each mask should have shape [H, W]
                if mask.dim() != 2:
                    raise ValueError(
                        f"Expected ref_mask to have 2 dimensions [H, W], but got shape {mask.shape}"
                    )

                # Convert the mask to a float tensor and add it to the tensor
                ref_masks_tensor[mask_idx, video_idx] = mask.to(torch.float)

        # Create BatchedVideoDatapoint with ref_masks
        return BatchedVideoDatapointSDF(
            img_batch=img_batch,
            obj_to_frame_idx=obj_to_frame_idx,
            masks=masks,
            metadata=BatchedVideoMetaData(
                unique_objects_identifier=objects_identifier,
                frame_orig_size=frame_orig_size,
            ),
            dict_key=dict_key,
            ref_masks=ref_masks_tensor,
            batch_size=[T],
        )
    else:
        # Create BatchedVideoDatapoint without ref_masks
        return BatchedVideoDatapointSDF(
            img_batch=img_batch,
            obj_to_frame_idx=obj_to_frame_idx,
            masks=masks,
            metadata=BatchedVideoMetaData(
                unique_objects_identifier=objects_identifier,
                frame_orig_size=frame_orig_size,
            ),
            dict_key=dict_key,
            batch_size=[T],
        )
