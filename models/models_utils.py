import os
import random
from typing import Callable
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from utils.types import Data, ClientsData


class PDLDataSet(torch.utils.data.Dataset):
    def __init__(self, data: Data, mode: str = 'normal', num_class: int = 10) -> None:
        self.X = torch.from_numpy(data[0])
        y = torch.from_numpy(data[1])
        if mode == 'normal':
            self.y = y
        elif mode == 'randomized':
            # @todo check
            self.y = torch.from_numpy(random.sample(data[1].tolist(), data[1].shape[0]))
        elif mode == 'flipped':
            # @todo check
            label_shift = random.randint(1, num_class - 1)
            self.y = torch.from_numpy(list((data[1] + label_shift) % num_class))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = self.X[idx, :, :, :]
        label = self.y[idx]
        return img, label


class SharedData(torch.utils.data.Dataset):
    def __init__(self, X: Tensor, Y: Tensor) -> None:
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.size(dim=0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        img = self.X[idx, :, :, :]
        label = self.Y[idx]
        return img, label, idx


# @todo better way to do that
def dl_to_sampler(dl) -> Callable[[], Tensor]:
    dl_iter = iter(dl)

    def sample() -> Tensor:
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)

    return sample


def transform_ref_data(ref_data: ClientsData) -> Tuple[Tensor, Tensor]:
    ref_X = np.concatenate(list(map(lambda x: x[0], ref_data)))
    ref_y = np.concatenate(list(map(lambda x: x[1], ref_data)))
    return torch.from_numpy(ref_X), torch.from_numpy(ref_y)
