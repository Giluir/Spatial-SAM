"""Native volume batches used by SDF regression."""

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class VolumePatch:
    """One image and SDF patch, each in ``[C, D, H, W]`` order."""

    image: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class VolumeBatch:
    """A batch of image and SDF tensors in ``[B, C, D, H, W]`` order."""

    images: torch.Tensor
    targets: torch.Tensor
    loss_key: str

    def to(self, device, non_blocking=False):
        return replace(
            self,
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self):
        return replace(
            self, images=self.images.pin_memory(), targets=self.targets.pin_memory()
        )


def collate_volume_patches(patches: list[VolumePatch], dict_key: str) -> VolumeBatch:
    """Stack native patches without converting depth slices to video objects."""
    return VolumeBatch(
        images=torch.stack([patch.image for patch in patches]),
        targets=torch.stack([patch.target for patch in patches]),
        loss_key=dict_key,
    )
