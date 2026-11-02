"""The 3D signed-distance regression network used by Spatial-SAM.

Adapted from pytorch3dunet. See licenses/PyTorch3DUNet.txt at the repository root.
Module names preserve compatibility with trained checkpoint keys.
"""

from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _single_conv(
    in_channels: int,
    out_channels: int,
    order: str,
    num_groups: int,
) -> nn.Sequential:
    """Build a convolution block with the original operation order and names."""
    if "c" not in order or order[0] in "rle":
        raise ValueError(f"Invalid convolution order: {order!r}")
    layers = OrderedDict()
    for index, operation in enumerate(order):
        channels = in_channels if index < order.index("c") else out_channels
        if operation == "c":
            layers["conv"] = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias="g" not in order and "b" not in order,
            )
        elif operation == "g":
            groups = 1 if channels < num_groups else num_groups
            if channels % groups:
                raise ValueError(
                    f"{channels} channels are not divisible by {groups} groups"
                )
            layers["groupnorm"] = nn.GroupNorm(groups, channels)
        elif operation == "b":
            layers["batchnorm"] = nn.BatchNorm3d(channels)
        elif operation == "r":
            layers["ReLU"] = nn.ReLU(inplace=True)
        elif operation == "l":
            layers["LeakyReLU"] = nn.LeakyReLU(inplace=True)
        elif operation == "e":
            layers["ELU"] = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported convolution operation: {operation!r}")
    return nn.Sequential(layers)


def _double_conv(
    in_channels: int,
    out_channels: int,
    *,
    encoder: bool,
    order: str,
    num_groups: int,
) -> nn.Sequential:
    """Preserve the narrower first convolution in each encoder block."""
    middle_channels = max(in_channels, out_channels // 2) if encoder else out_channels
    return nn.Sequential(
        OrderedDict(
            SingleConv1=_single_conv(in_channels, middle_channels, order, num_groups),
            SingleConv2=_single_conv(middle_channels, out_channels, order, num_groups),
        )
    )


class _Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, *, pool, order, num_groups):
        super().__init__()
        self.pooling = nn.MaxPool3d(2) if pool else None
        self.basic_module = _double_conv(
            in_channels, out_channels, encoder=True, order=order, num_groups=num_groups
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if self.pooling is not None:
            volume = self.pooling(volume)
        return self.basic_module(volume)


class _Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, *, order, num_groups):
        super().__init__()
        self.basic_module = _double_conv(
            in_channels, out_channels, encoder=False, order=order, num_groups=num_groups
        )

    def forward(self, skip: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        # Resize to the skip shape to support odd spatial dimensions.
        volume = F.interpolate(volume, size=skip.shape[2:], mode="nearest")
        return self.basic_module(torch.cat((skip, volume), dim=1))


class SDFUNet3D(nn.Module):
    """Predict a signed distance field from a batch of 3D image patches.

    Args:
        in_channels: Number of image channels.
        out_channels: Number of output distance fields.
        f_maps: Encoder widths, or the initial width doubled at each level.
        layer_order: Order of convolution, normalization, and activation.
        num_groups: Number of groups used by group normalization.
        num_levels: Number of encoder levels when ``f_maps`` is an integer.
        final_sigmoid: Choose sigmoid instead of softmax for segmentation.
        is_segmentation: Apply the final activation during evaluation only.

    Inputs and outputs use the layout ``[batch, channels, depth, height, width]``.
    SDF regression leaves the output unbounded in both training and evaluation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        f_maps: int | Sequence[int] = (32, 64, 128, 256),
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        final_sigmoid: bool = True,
        is_segmentation: bool = False,
    ):
        super().__init__()
        widths = (
            [f_maps * 2**level for level in range(num_levels)]
            if isinstance(f_maps, int)
            else list(f_maps)
        )
        if len(widths) < 2 or any(width <= 0 for width in widths):
            raise ValueError("f_maps must contain at least two positive widths")
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        self.encoders = nn.ModuleList(
            _Encoder(
                in_channels if level == 0 else widths[level - 1],
                width,
                pool=level > 0,
                order=layer_order,
                num_groups=num_groups,
            )
            for level, width in enumerate(widths)
        )
        self.decoders = nn.ModuleList(
            _Decoder(
                widths[level] + widths[level - 1],
                widths[level - 1],
                order=layer_order,
                num_groups=num_groups,
            )
            for level in range(len(widths) - 1, 0, -1)
        )
        self.final_conv = nn.Conv3d(widths[0], out_channels, kernel_size=1)
        self.final_activation = (
            (nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1))
            if is_segmentation
            else None
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], got {tuple(volume.shape)}")
        features = []
        for encoder in self.encoders:
            volume = encoder(volume)
            features.append(volume)
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            volume = decoder(skip, volume)
        volume = self.final_conv(volume)
        if not self.training and self.final_activation is not None:
            volume = self.final_activation(volume)
        return volume
