"""SDF regression loss on native ``[B, C, D, H, W]`` tensors."""

from collections.abc import Sequence

import torch
from torch import nn


class MSEWithEikonalLoss(nn.Module):
    """Combine mean squared error with the original masked Eikonal penalty."""

    def __init__(
        self,
        lambda_eikonal: float = 0.1,
        voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
        truncation_threshold: float = 0.95,
    ):
        super().__init__()
        self.lambda_eikonal = lambda_eikonal
        self.truncation_threshold = truncation_threshold
        self.register_buffer(
            "voxel_size", torch.tensor(voxel_size, dtype=torch.float32)
        )

    def forward(self, prediction, target, weight=None):
        if prediction.ndim != 5 or prediction.shape != target.shape:
            raise ValueError(
                "SDF prediction and target must have identical [B, C, D, H, W] shapes"
            )
        prediction, target = prediction.float(), target.float()
        grad_z, grad_y, grad_x = torch.gradient(prediction, dim=(-3, -2, -1))
        gradients = torch.stack(
            [
                grad_x / self.voxel_size[0],
                grad_y / self.voxel_size[1],
                grad_z / self.voxel_size[2],
            ],
            dim=-1,
        )
        if weight is None:
            mse = nn.functional.mse_loss(prediction, target)
        else:
            weights = ((0.5 - weight.float()) * 2) ** 2
            mse = torch.mean(weights * (prediction - target) ** 2)
        threshold = self.truncation_threshold * torch.max(torch.abs(target))
        mask = torch.abs(target) < threshold
        magnitude = torch.norm(gradients, p=2, dim=-1)[mask]
        eikonal = (
            torch.mean((magnitude - 1) ** 2)
            if magnitude.numel()
            else torch.tensor(0.0, device=prediction.device, requires_grad=True)
        )
        return mse + self.lambda_eikonal * eikonal
