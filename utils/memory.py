import bz2
import datetime
import os
import pickle
import time
from typing import Tuple

import ujson

from utils.folders import get_config_path, get_save_train_test_ref_path, __get_dataset_path
from utils.types import ClientsData, Config, DatasetName


def __save_config(config: Config, path: str) -> None:
    with open(path, 'w') as f:
        ujson.dump(config.__dict__, f)


def __save_data(data: ClientsData, path: str) -> None:
    with bz2.open(path, 'wb') as f:
        pickle.dump(data, f)


def __load_data(path: str) -> ClientsData:
    with bz2.open(path, 'rb') as f:
        return pickle.load(f)


def load_dataset(dataset_name: DatasetName) -> Tuple[ClientsData, ClientsData, ClientsData]:
    diff_config = os.listdir(f'{__get_dataset_path(dataset_name)}save/')
    print(f'folders for {dataset_name}:')
    for idx, folder in enumerate(diff_config):
        print(f'{idx}: {folder}')
    choice = int(input())
    path = f'{__get_dataset_path(dataset_name)}save/{diff_config[choice]}/'
    train_path, test_path, ref_path = f'{path}train.bz2', f'{path}test.bz2', f'{path}ref.bz2'
    return __load_data(train_path), __load_data(test_path), __load_data(ref_path)


def save_dataset(train_data: ClientsData, test_data: ClientsData, ref_data: ClientsData, config: Config) -> None:
    print('------ Saving to disk ------')
    # perf measurement
    time_start = time.perf_counter()

    train_path, test_path, ref_path = get_save_train_test_ref_path(config)
    __save_data(train_data, train_path)
    __save_data(test_data, test_path)
    __save_data(ref_data, ref_path)
    __save_config(config, get_config_path(config))

    time_end = time.perf_counter()
    print('-' * 40)
    print(
        'Dataset saved - Time elapsed: ',
        str(datetime.timedelta(seconds=time_end - time_start))
    )
    print('------ Finish saving to disk ------')
