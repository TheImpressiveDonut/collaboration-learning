from utils.types import DatasetName, SettingName


class UnknownDatasetNameException(Exception):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(f'Dataset name {dataset_name} not valid, names accepted: '
                         f'{", ".join([j.__str__() for j in DatasetName])}')

class UnknownSettingNameException(Exception):
    def __init__(self, setting_name: str) -> None:
        super().__init__(f'Dataset name {setting_name} not valid, names accepted: '
                         f'{", ".join([j.__str__() for j in SettingName])}')


class UnknownPartitionArgumentException(Exception):
    def __int__(self, partition_name: str) -> None:
        super().__init__(f'Partition name {partition_name} not valid, names accepted: pat, dir')


class UnknownSplitDataModeException(Exception):
    def __int__(self, mode: str) -> None:
        super().__init__(f'Mode name {mode} not valid, names accepted: by-mode, by-number')
