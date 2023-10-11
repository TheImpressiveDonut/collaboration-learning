from enum import StrEnum
from typing import Type

from utils.types import DatasetName, SettingName, MetricName


class UnknownNameCustomEnumException(Exception):

    def __init__(self, unknown_name: str, enum: Type[StrEnum]) -> None:
        super().__init__(f'{enum.__class__.__name__} {unknown_name} not valid, names accepted: '
                         f'{", ".join([j.__str__() for j in enum])}')


class UnknownPartitionArgumentException(Exception):
    def __int__(self, partition_name: str) -> None:
        super().__init__(f'Partition name {partition_name} not valid, names accepted: pat, dir')


class UnknownSplitDataModeException(Exception):
    def __int__(self, mode: str) -> None:
        super().__init__(f'Mode name {mode} not valid, names accepted: by-mode, by-number')
