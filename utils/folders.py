import os
from typing import Tuple

from utils.mlo_folders import get_mlo_dir_dataset_path
from utils.types import DatasetName, Config


def __create_folder(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def __get_dataset_path(dataset_name: DatasetName) -> True:
    return f'{get_mlo_dir_dataset_path()}{dataset_name}/'


def __get_path_new_dir(dataset_name: DatasetName, dir: str, create_dir: bool = True) -> str:
    path = f'{__get_dataset_path(dataset_name)}{dir}/'
    if create_dir:
        __create_folder(path)
    return path


def __get_save_path(config: Config, create_dir: bool = True) -> str:
    return __get_path_new_dir(config.dataset_name, f'save/{config.hash()}/', create_dir)


def get_config_path(config: Config) -> str:
    return f'{__get_save_path(config, create_dir=True)}config.json'


def get_save_train_test_ref_path(config: Config) -> Tuple[str, str, str]:
    path = __get_save_path(config, create_dir=True)
    return f'{path}train.bz2', f'{path}test.bz2', f'{path}ref.bz2'


def get_raw_path(dataset_name: DatasetName, create_dir: bool = True) -> str:
    return __get_path_new_dir(dataset_name, 'raw', create_dir)


def check_if_config_exist(config: Config) -> bool:
    return os.path.exists(__get_save_path(config, create_dir=False))
