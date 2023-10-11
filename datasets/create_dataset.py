import argparse
import datetime
import random
import time
from typing import Tuple

import numpy as np
import torchvision

from datasets.generation.cifar_x import cifar_x
from datasets.generation.fed_isic import fed_isic
from datasets.parser import get_parser
from utils.exceptions import UnknownNameCustomEnumException
from utils.folders import check_if_config_exist
from utils.memory import save_dataset
from utils.types import ClientsData, ClientsDataStatistics, Config, DatasetName

# @todo shouldn't we try with different seeds for real experiments ?
random.seed(1)
np.random.seed(1)


def create_dataset(config: Config) -> Tuple[ClientsData, ClientsData, ClientsData, ClientsDataStatistics]:
    match config.dataset_name:
        case DatasetName.cifar10:
            return cifar_x(dataset=torchvision.datasets.CIFAR10, config=config)
        case DatasetName.cifar100:
            return cifar_x(dataset=torchvision.datasets.CIFAR100, config=config)
        case DatasetName.fed_isic:
            # num_classes = 8, num_clients = 6
            return fed_isic(config)
        case _:
            raise UnknownNameCustomEnumException(config.dataset_name, DatasetName)


print('------ Generating dataset ------')
args = get_parser().parse_args().__dict__
config_dataset = Config(
    dataset_name=args['dataset_name'], num_clients=args['num_clients'], num_classes=args['num_classes'],
    class_per_client=args['class_per_client'], niid=args['niid'], ref=args['ref'], balance=args['balance'],
    partition=args['partition'], alpha=args['alpha'], batch_size=args['batch_size'],
    train_size=args['train_size']
)
print(config_dataset)

if check_if_config_exist(config_dataset):
    print('-' * 6, ' Dataset already generated ', '-' * 6)
else:
    # perf measurement
    time_start = time.perf_counter()

    train_data, test_data, ref_data, statistic = create_dataset(config_dataset)
    save_dataset(train_data, test_data, ref_data, config_dataset)

    time_end = time.perf_counter()
    print('-' * 40)
    print(
        'Dataset generated - Time elapsed: ',
        str(datetime.timedelta(seconds=time_end - time_start))
    )
print('-' * 6, ' End Generating dataset ', '-' * 6)
