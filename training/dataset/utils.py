# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM2.txt file in the root directory of this source tree.

"""Some wrapping utilities extended from pytorch's to support repeat factor sampling in particular"""

from typing import Iterable

import torch
from torch.utils.data import (
    ConcatDataset as TorchConcatDataset,
    Dataset,
    Subset as TorchSubset,
)


class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__(datasets)

        self.repeat_factors = torch.cat([d.repeat_factors for d in datasets])

    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            if hasattr(dataset, "epoch"):
                dataset.epoch = epoch
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


class Subset(TorchSubset):
    def __init__(self, dataset, indices) -> None:
        super(Subset, self).__init__(dataset, indices)

        self.repeat_factors = dataset.repeat_factors[indices]
        assert len(indices) == len(self.repeat_factors)


# Adapted from Detectron2
