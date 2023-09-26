import hashlib
from enum import StrEnum, auto
from typing import Tuple, List, Dict

import numpy as np

from client.client import Client

Data = Tuple[np.ndarray, np.ndarray]
ClientsData = List[Data]
Id = int
Clients = Dict[Id, Client]
DataStatistics = List[Tuple[int, int]]
ClientsDataStatistics = List[DataStatistics]


class DatasetName(StrEnum):
    cifar10 = auto()
    cifar100 = auto()
    fed_isic = auto()


class SettingName(StrEnum):
    normal = auto()
    two_sets = auto()
    evil = auto()


class Config(object):

    def __init__(self,
                 dataset_name: DatasetName, num_clients: int, num_classes: int, class_per_client: int,
                 niid: bool, ref: bool, balance: bool, partition: str, alpha: float, batch_size: int,
                 train_size: float,
                 ) -> None:
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.class_per_client = class_per_client
        self.niid = niid
        self.ref = ref
        self.balance = balance
        self.partition = partition
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_size = train_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return (self.dataset_name == other.dataset_name and
                self.num_clients == other.num_clients and
                self.num_classes == other.num_classes and
                self.class_per_client == other.class_per_client and
                self.niid == other.niid and
                self.ref == other.ref and
                self.balance == other.balance and
                self.partition == other.partition and
                self.alpha == other.alpha and
                self.batch_size == other.batch_size and
                self.train_size == other.train_size)

    def hash(self) -> str:
        hash = hashlib.sha256()
        hash.update(
            (f'{self.dataset_name},'
             f'{self.num_clients},'
             f'{self.num_classes},'
             f'{self.class_per_client},'
             f'{self.niid},'
             f'{self.ref},'
             f'{self.balance},'
             f'{self.alpha},'
             f'{self.batch_size},'
             f'{self.train_size}').encode()
        )
        return hash.hexdigest()

    def __str__(self) -> str:
        return (
            f'Config:\n'
            f'\t{self.dataset_name}\n'
            f'\t{self.num_clients}\n'
            f'\t{self.num_classes}\n'
            f'\t{self.class_per_client}\n'
            f'\t{self.niid}\n'
            f'\t{self.ref}\n'
            f'\t{self.balance}\n'
            f'\t{self.alpha}\n'
            f'\t{self.batch_size}\n'
            f'\t{self.train_size}\n'
        )
