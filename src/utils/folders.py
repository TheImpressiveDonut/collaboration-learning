import os
from argparse import Namespace
from typing import Tuple

from .mlo_folders import get_mlo_dir_dataset_path
from .types import DatasetName


def __create_folder(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def __get_dataset_path(dataset_name: DatasetName) -> True:
    return f'{get_mlo_dir_dataset_path()}data/{dataset_name}/'


def __get_path_new_dir(dataset_name: DatasetName, dir: str, create_dir: bool = True) -> str:
    path = f'{__get_dataset_path(dataset_name)}{dir}/'
    if create_dir:
        __create_folder(path)
    return path


def __get_save_path(args: Namespace, create_dir: bool = True) -> str:
    return __get_path_new_dir(args.dataset_name, args.__repr__(), create_dir)


def get_config_path(args: Namespace) -> str:
    return f'{__get_save_path(args, create_dir=True)}config.json'


def get_save_train_test_ref_stats_path(args: Namespace) -> Tuple[str, str, str, str]:
    path = __get_save_path(args, create_dir=True)
    return f'{path}train.bz2', f'{path}test.bz2', f'{path}ref.bz2', f'{path}stats.bz2'


def get_raw_path(dataset_name: DatasetName, create_dir: bool = True) -> str:
    return __get_path_new_dir(dataset_name, 'raw', create_dir)


def check_if_config_exist(args: Namespace) -> bool:
    return os.path.exists(__get_save_path(args, create_dir=False))
