"""Task-specific forward passes consumed by the shared training loop."""

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from training.batch import VolumeBatch
from training.utils.data_utils import BatchedVideoDatapoint


@dataclass(frozen=True)
class StepInputs:
    """Outputs and supervision needed by the shared loss and logging code."""

    predictions: Any
    targets: torch.Tensor
    loss_key: str
    batch_size: int
    metadata: Any = None


class SAMSegmentationTask:
    """Preserve SAM2's video forward pass and frame-based loss accounting."""

    def __call__(self, model: nn.Module, batch: BatchedVideoDatapoint) -> StepInputs:
        return StepInputs(
            model(batch),
            batch.masks,
            batch.dict_key,
            len(batch.img_batch),
            batch.metadata,
        )

    def checkpoint_state(self, model):
        """Preserve the model's independent prompt-sampling generator."""
        return {"prompt_rng": deepcopy(model.rng.bit_generator.state)}

    def restore_checkpoint_state(self, model, state):
        """Restore prompt choices when resuming within a training stage."""
        model.rng.bit_generator.state = state["prompt_rng"]


class SDFRegressionTask:
    """Regress distance fields directly from native volume tensors."""

    def __call__(self, model: nn.Module, batch: VolumeBatch) -> StepInputs:
        return StepInputs(
            model(batch.images), batch.targets, batch.loss_key, len(batch.images)
        )

    def checkpoint_state(self, model):
        """SDF randomness belongs to the dataset and the global training state."""
        return {}

    def restore_checkpoint_state(self, model, state):
        """Reject unexpected task state instead of silently ignoring it."""
        if state:
            raise ValueError("SDF regression has no model-local random state")
