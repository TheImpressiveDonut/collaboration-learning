import bz2
import pickle
from argparse import Namespace
from typing import Any

import torch
import ujson

from src.utils.folders import get_config_path, get_save_train_test_ref_path, __create_folder
from src.utils.mlo_folders import get_mlo_dir_res_path
from src.utils.types import ClientsData, Dataset


def __load_config(path: str) -> Any:
    with open(path, 'r') as f:
        return ujson.load(f)


def __save_config(args: Namespace, path: str) -> None:
    with open(path, 'w') as f:
        ujson.dump(args.__dict__, f)


def __save_data(data: ClientsData, path: str) -> None:
    with bz2.open(path, 'wb') as f:
        pickle.dump(data, f)


def __load_data(path: str) -> ClientsData:
    with bz2.open(path, 'rb') as f:
        return pickle.load(f)


def load_dataset(args: Namespace) -> Dataset:
    train_path, test_path, ref_path = get_save_train_test_ref_path(args)
    return __load_data(train_path), __load_data(test_path), __load_data(ref_path)


def save_dataset(train_data: ClientsData, test_data: ClientsData, ref_data: ClientsData, args: Namespace) -> None:
    train_path, test_path, ref_path = get_save_train_test_ref_path(args)
    __save_data(train_data, train_path)
    __save_data(test_data, test_path)
    __save_data(ref_data, ref_path)
    __save_config(args, get_config_path(args))


def save_results(trust_weights, test_accuracies, ref_accuracies, dataset_name: str, experiment_name: str) -> None:
    path = f'{get_mlo_dir_res_path()}{dataset_name}/{experiment_name}/'
    __create_folder(path)
    torch.save(trust_weights, f'{path}trust.pt')
    torch.save(test_accuracies, f'{path}test.pt')
    torch.save(ref_accuracies, f'{path}ref.pt')
