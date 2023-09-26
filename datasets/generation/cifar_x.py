# codes adapted from https://github.com/TsingZ0/PFL-Non-IID/blob/master/dataset/generate_cifar10.py

from typing import Type, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

from utils.data import clients_split, train_test_ref_split
from utils.folders import get_raw_path
from utils.types import ClientsData, ClientsDataStatistics, Config


def cifar_x(dataset: Type[CIFAR10 | CIFAR100], config: Config) -> Tuple[
    ClientsData, ClientsData, ClientsData, ClientsDataStatistics
]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Get CifarX data
    raw_path = get_raw_path(config.dataset_name)
    train_dataset = dataset(
        root=raw_path, train=True, download=True, transform=transform
    )
    test_dataset = dataset(
        root=raw_path, train=False, download=False, transform=transform
    )

    # We use a DataLoader because we want to apply the Compose transform on each data sample
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.data.shape[0], shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.data.shape[0], shuffle=False)

    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))
    dataset_image = np.concatenate((train_data.cpu().detach().numpy(), test_data.cpu().detach().numpy()))
    dataset_label = np.concatenate((train_labels.cpu().detach().numpy(), test_labels.cpu().detach().numpy()))

    clients_data, statistic = clients_split(data=(dataset_image, dataset_label), config=config)
    train_data, test_data, ref_data = train_test_ref_split(clients_data, config)
    return train_data, test_data, ref_data, statistic
