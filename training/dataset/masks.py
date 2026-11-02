# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.


import torch


class VolumeSegmentLoader:
    def __init__(self, masks, single_object_mode=True, to_binary=True):
        if single_object_mode and to_binary:
            # In single object mode, we see whole mask as a single object
            binary_masks = masks > 0
        else:
            binary_masks = masks

        self.obj_id = 1 if single_object_mode else None
        self.single_object_mode = single_object_mode
        self.masks = torch.from_numpy(binary_masks)

    def load(self, frame_id):
        if self.single_object_mode:
            return {self.obj_id: self.masks[frame_id]}
        else:
            # Get the mask for the current frame
            frame_mask = self.masks[frame_id]

            # Find all unique values in the mask (excluding 0 which is background)
            unique_values = torch.unique(frame_mask)
            unique_values = unique_values[unique_values > 0]

            # Create a dictionary mapping each object ID to its binary mask
            binary_segments = {}
            for obj_id in unique_values:
                # Create a binary mask for this object ID
                binary_mask = frame_mask == obj_id
                binary_segments[obj_id.item()] = binary_mask

            return binary_segments


class RefMaskLoader:
    def __init__(self, ref_masks, to_logits=False):
        self.ref_masks = ref_masks
        self.ref_masks = torch.from_numpy(self.ref_masks)
        if to_logits:
            self.ref_masks = self.binary_to_logits(self.ref_masks)

    def binary_to_logits(self, mask: torch.Tensor):
        return mask.float() * 20.0 - 10.0

    def load(self, frame_ids):
        ref_masks = []
        for frame_id in frame_ids:
            ref_masks.append(self.ref_masks[frame_id])

        return ref_masks
