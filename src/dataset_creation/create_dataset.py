import datetime
import random
import time
from argparse import Namespace
from typing import Tuple

import numpy as np
import torchvision

from src.dataset_creation.generation.cifar_x import cifar_x
from src.dataset_creation.generation.fed_isic import fed_isic
from src.dataset_creation.parser import get_parser
from src.utils.exceptions import UnknownNameCustomEnumException
from src.utils.folders import check_if_config_exist
from src.utils.memory import save_dataset, load_dataset
from src.utils.types import ClientsData, ClientsDataStatistics, DatasetName, Dataset

random.seed(1)
np.random.seed(1)


def create_dataset(args: Namespace) -> Tuple[ClientsData, ClientsData, ClientsData, ClientsDataStatistics]:
    match args.dataset_name:
        case DatasetName.cifar10:
            return cifar_x(dataset=torchvision.datasets.CIFAR10, args=args)
        case DatasetName.cifar100:
            return cifar_x(dataset=torchvision.datasets.CIFAR100, args=args)
        case DatasetName.fed_isic:
            # num_classes = 8, num_clients = 6
            return fed_isic(args)
        case _:
            raise UnknownNameCustomEnumException(args.dataset_name, DatasetName)


def main(args: Namespace) -> Dataset:
    print('-' * 5, ' Generating dataset ', '-' * 5)
    print(args)

    if check_if_config_exist(args):
        print('-' * 5, ' Dataset already generated ', '-' * 5)
        return load_dataset(args)
    else:
        train_data, test_data, ref_data, statistic = create_dataset(args)
        save_dataset(train_data, test_data, ref_data, args)
        print('-' * 5, ' Dataset generated ', '-' * 5)
        return train_data, test_data, ref_data

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)