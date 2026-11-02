"""Metric-driven scheduling and stopping rules for the shared training loop."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UpdateDecision:
    """Actions to take after a completed optimizer update."""

    publish_weights: bool
    stop: bool


class EpochLossPlateau:
    """Step a plateau scheduler periodically using the current epoch's mean loss.

    The loss average resets at the epoch boundary; the update counter does not.
    Published weights correspond to scheduling boundaries, preserving the configured
    UNet's handoff checkpoint independently from recovery checkpoints.
    """

    def __init__(
        self,
        optimizer,
        interval=2000,
        max_updates=100001,
        min_lr=1e-6,
        factor=0.2,
        patience=8,
    ):
        if interval <= 0 or max_updates <= 0:
            raise ValueError("Scheduling interval and update limit must be positive")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )
        self.interval = interval
        self.max_updates = max_updates
        self.min_lr = min_lr
        self.updates = 0
        self.begin_epoch()

    def begin_epoch(self):
        """Reset the loss average used by the original per-epoch training loop."""
        self.loss_sum = 0.0
        self.sample_count = 0

    def after_update(self, loss: float, batch_size: int) -> UpdateDecision:
        """Schedule after updating weights, then evaluate the stopping criteria."""
        self.loss_sum += loss * batch_size
        self.sample_count += batch_size
        self.updates += 1
        publish = self.updates % self.interval == 0
        if publish:
            self.scheduler.step(self.loss_sum / self.sample_count)
        stop = (
            self.updates >= self.max_updates
            or self.scheduler.optimizer.param_groups[0]["lr"] < self.min_lr
        )
        return UpdateDecision(publish, stop)

    def state_dict(self):
        """Include the scheduler and aggregation state for interrupted stages."""
        return {
            "scheduler": self.scheduler.state_dict(),
            "updates": self.updates,
            "loss_sum": self.loss_sum,
            "sample_count": self.sample_count,
        }

    def load_state_dict(self, state):
        """Restore state after the optimizer has been restored."""
        self.scheduler.load_state_dict(state["scheduler"])
        self.updates = state["updates"]
        self.loss_sum = state["loss_sum"]
        self.sample_count = state["sample_count"]
