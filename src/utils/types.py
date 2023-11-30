import hashlib
from enum import StrEnum, auto
from typing import Tuple, List

import numpy as np

Data = Tuple[np.ndarray, np.ndarray]
ClientsData = List[Data]
Dataset = Tuple[ClientsData, ClientsData, ClientsData]
DataStatistics = List[Tuple[int, int]]
ClientsDataStatistics = List[DataStatistics]


class TrustName(StrEnum):
    static = auto()
    dynamic = auto()
    naive = auto()


class ConsensusName(StrEnum):
    soft_assignment = auto()
    majority_vote = auto()


class SimMeasureName(StrEnum):
    cosine = auto()
    true_label = auto()


class ModelName(StrEnum):
    resnet = auto()
    cnn = auto()
    fnn = auto()
    efficient_net = auto()


class ModeName(StrEnum):
    normal = auto()
    flipped = auto()


class DatasetName(StrEnum):
    cifar10 = auto()
    cifar100 = auto()
    fed_isic = auto()


class SettingName(StrEnum):
    normal = auto()
    two_sets = auto()
    evil = auto()


class MetricName(StrEnum):
    acc = auto()
    bacc = auto()