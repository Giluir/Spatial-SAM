# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List


MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class AnnotatedSliceSampler(VOSSampler):
    def __init__(self, num_frames, max_num_objects, reverse_time_prob=0.0, gt_prob=0.5):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        self.gt_prob = gt_prob

    def sample(self, video, segment_loader, epoch=None, few_masks_info=None):
        """
        Sample frames with gt_prob probability of centering around few_masks frames

        Args:
            video: VOSVideo object
            segment_loader: Segment loader
            epoch: Current epoch (unused)
            few_masks_info: List of frame indices that have few_masks

        Returns:
            SampledFramesAndObjects containing sampled frames and object IDs
        """
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )

            # Decide whether to use GT-based sampling or random sampling
            use_gt_sampling = (
                random.uniform(0, 1) < self.gt_prob and few_masks_info is not None
            )

            if use_gt_sampling:
                # GT-based sampling: center around few_masks frames
                frames = self._sample_around_gt_frames(video, few_masks_info)
            else:
                # Random sampling (uniform sampling)
                frames = self._sample_random_frames(video)

            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            for object_id, segment in segment_loader.load(frames[0].frame_idx).items():
                if segment.sum():
                    visible_object_ids.append(object_id)

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)

    def _sample_around_gt_frames(self, video, few_masks_info):
        """
        Sample frames centered around few_masks frames
        """
        # few_masks_info is now a list of frame indices (z-direction only)
        all_gt_frames = few_masks_info

        if not all_gt_frames:
            # Fallback to random sampling if no GT frames
            return self._sample_random_frames(video)

        # Choose a random GT frame as center
        center_frame_idx = random.choice(all_gt_frames)

        # Calculate start and end indices for sampling
        half_frames = self.num_frames // 2
        start_idx = max(0, center_frame_idx - half_frames)
        end_idx = min(len(video.frames), center_frame_idx + half_frames + 1)

        # Adjust if we don't have enough frames
        if end_idx - start_idx < self.num_frames:
            if start_idx == 0:
                end_idx = min(len(video.frames), start_idx + self.num_frames)
            else:
                start_idx = max(0, end_idx - self.num_frames)

        # Sample the frames
        sampled_indices = list(
            range(start_idx, min(start_idx + self.num_frames, len(video.frames)))
        )
        frames = [video.frames[i] for i in sampled_indices]

        return frames

    def _sample_random_frames(self, video):
        """
        Sample frames randomly (uniform sampling)
        """
        start = random.randrange(0, len(video.frames) - self.num_frames + 1)
        frames = [video.frames[start + step] for step in range(self.num_frames)]
        return frames
