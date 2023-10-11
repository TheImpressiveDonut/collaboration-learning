from argparse import Namespace
from typing import Tuple, Dict

import numpy as np

from client.client import Client
from utils.exceptions import UnknownNameCustomEnumException
from utils.types import Clients, DatasetName, SettingName, ClientsData, ModeName, ModelName


def setting_selection(setting: SettingName, dataset_name: DatasetName, num_clients: int
                      ) -> Dict[int, Tuple[ModelName, ModeName]]:
    clients = {}
    match setting:
        case SettingName.normal:
            match dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    for i in range(num_clients):
                        clients[i] = (ModelName.resnet, ModeName.normal)
                case DatasetName.fed_isic:
                    for i in range(num_clients):
                        clients[i] = (ModelName.efficient_net, ModeName.normal)
                case _:
                    raise UnknownNameCustomEnumException(dataset_name, DatasetName)

        case SettingName.two_sets:
            middle = num_clients // 2
            match dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    for i in range(middle):
                        clients[i] = (ModelName.resnet, ModeName.normal)
                    for i in range(middle, num_clients):
                        clients[i] = (ModelName.fnn, ModeName.normal)
                case DatasetName.fed_isic:
                    for i in range(middle):
                        clients[i] = (ModelName.efficient_net, ModeName.normal)
                    for i in range(middle, num_clients):
                        clients[i] = (ModelName.fnn, ModeName.normal)
                case _:
                    raise UnknownNameCustomEnumException(dataset_name, DatasetName)

        case SettingName.evil:
            client_ids = np.arange(num_clients)  # @todo maybe scale num of evil to num of clients %
            match dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    evil_idx = np.array([2, 9])
                    print('evil worker:', evil_idx)
                    normal_idx = [id for id in client_ids if id not in evil_idx]
                    for i in evil_idx:
                        clients[i] = (ModelName.resnet, ModeName.flipped)
                    for i in normal_idx:
                        clients[i] = (ModelName.resnet, ModeName.normal)
                case DatasetName.fed_isic:
                    evil_idx = np.array([1, 2])
                    print('evil worker:', evil_idx)
                    normal_idx = [id for id in client_ids if id not in evil_idx]
                    for i in evil_idx:
                        clients[i] = (ModelName.efficient_net, ModeName.flipped)
                    for i in normal_idx:
                        clients[i] = (ModelName.efficient_net, ModeName.normal)
                case _:
                    raise UnknownNameCustomEnumException(dataset_name, DatasetName)
        case _:
            raise UnknownNameCustomEnumException(setting, SettingName)

    return clients
