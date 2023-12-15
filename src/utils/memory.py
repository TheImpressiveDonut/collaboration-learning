import bz2
import pickle
import ujson
from argparse import Namespace
from typing import Any, Tuple

from .folders import get_config_path, get_save_train_test_ref_stats_path
from .types import ClientsData, Dataset, ClientsDataStatistics


def __load_config(path: str) -> Any:
    with open(path, 'r') as f:
        return ujson.load(f)


def __save_config(args: Namespace, path: str) -> None:
    with open(path, 'w') as f:
        ujson.dump(args.__dict__, f)


def __save_data(data: ClientsData | ClientsDataStatistics, path: str) -> None:
    with bz2.open(path, 'wb') as f:
        pickle.dump(data, f)


def __load_data(path: str) -> ClientsData | ClientsDataStatistics:
    with bz2.open(path, 'rb') as f:
        return pickle.load(f)


def load_dataset(args: Namespace) -> Tuple[Dataset, ClientsDataStatistics]:
    train_path, test_path, ref_path, stats_path = get_save_train_test_ref_stats_path(args)
    return (__load_data(train_path), __load_data(test_path), __load_data(ref_path)), __load_data(stats_path)


def save_dataset(train_data: ClientsData, test_data: ClientsData, ref_data: ClientsData,
                 statistics: ClientsDataStatistics, args: Namespace) -> None:
    train_path, test_path, ref_path, stats_path = get_save_train_test_ref_stats_path(args)
    __save_data(train_data, train_path)
    __save_data(test_data, test_path)
    __save_data(ref_data, ref_path)
    __save_data(statistics, stats_path)
    __save_config(args, get_config_path(args))
