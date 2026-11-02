"""Loader policies for native tensors without video or mixture adapters."""

from torch.utils.data import DataLoader


class EpochDataLoader:
    """Keep PyTorch's ordinary random sampler and worker lifecycle across epochs."""

    def __init__(
        self,
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    def get_loader(self, epoch):
        """Let the dataset choose its epoch while preserving sampler semantics."""
        self.dataset.set_epoch(epoch)
        return self.loader

    def get_checkpoint_state(self):
        """Preserve main-process transform streams at completed epoch boundaries."""
        return self.dataset.get_checkpoint_state()

    def load_checkpoint_state(self, state):
        """Restore transform streams before creating the next loader iterator."""
        self.dataset.load_checkpoint_state(state)
