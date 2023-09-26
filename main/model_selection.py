from argparse import Namespace

import numpy as np

from client.client import Client
from utils.exceptions import UnknownSettingNameException, UnknownDatasetNameException
from utils.types import Clients, DatasetName, SettingName


def select_clients_model(args: Namespace) -> Clients:
    clients: Clients = {}
    match args.setting:
        case SettingName.normal:
            mode = 'normal'
            match args.dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    model_name = 'resnet'
                case DatasetName.fed_isic:
                    model_name = 'efficient-net'
                case _:
                    raise UnknownDatasetNameException(args.dataset_name)

        case SettingName.two_sets:
            mode = 'normal'
            match args.dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    for i in range(0, 5, 1):
                        clients[i] = Client(i, model_name='resnet', mode='normal', args=args)
                    for i in range(5, 10, 1):
                        clients[i] = Client(i, model_name='fnn', mode='normal', args=args)
                case DatasetName.fed_isic:
                    for i in range(0, 3, 1):
                        clients[i] = Client(i, model_name='efficient-net', mode='normal', args=args)
                    for i in range(3, 6, 1):
                        clients[i] = Client(i, model_name='fnn', mode='normal', args=args)
                case _:
                    raise UnknownDatasetNameException(args.dataset_name)

        case SettingName.evil:
            client_ids = np.arange(args.num_clients)
            match args.dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    evil_idx = np.array([2, 9])
                    print('evil worker:', evil_idx)
                    normal_idx = [id for id in client_ids if id not in evil_idx]
                    for i in evil_idx:
                        clients[i] = Client(i, model_name='resnet', mode='flipped', args=args)
                    for i in normal_idx:
                        clients[i] = Client(i, model_name='resnet', mode='normal', args=args)
                case DatasetName.fed_isic:
                    evil_idx = np.array([1, 2])
                    print('evil worker:', evil_idx)
                    normal_idx = [id for id in client_ids if id not in evil_idx]
                    for i in evil_idx:
                        clients[i] = Client(i, model_name='efficient-net', mode='flipped', args=args)
                    for i in normal_idx:
                        clients[i] = Client(i, model_name='efficient-net', mode='normal', args=args)
                case _:
                    raise UnknownDatasetNameException(args.dataset_name)
        case _:
            raise UnknownSettingNameException(args.setting)

    return clients
