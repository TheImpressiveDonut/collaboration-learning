import numpy as np
from enum import StrEnum, auto
from typing import Tuple, List

Data = Tuple[np.ndarray, np.ndarray]
ClientsData = List[Data]
Dataset = Tuple[ClientsData, ClientsData, ClientsData]
DataStatistics = List[Tuple[int, int]]
ClientsDataStatistics = List[DataStatistics]


class TrustName(StrEnum):
    none = auto()
    naive = auto()
    dynamic = auto()


class ModeName(StrEnum):
    normal = auto()
    flipped = auto()


class DatasetName(StrEnum):
    agnews = auto()
    multi_wikitext = auto()
