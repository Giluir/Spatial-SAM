# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

"""
Tensor conversion and channel expansion for SAM volume training.
"""


import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2


from training.utils.data_utils import VideoDatapoint


class ToTensor:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                if not isinstance(img.data, torch.Tensor):
                    img.data = F.to_tensor(img.data)
        return datapoint


class RepeatChannels:
    def __init__(self, num_channels=3):
        self.num_channels = num_channels

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if img.data.size(0) == 1:
                img.data = img.data.repeat(self.num_channels, 1, 1)
        return datapoint


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
