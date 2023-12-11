from argparse import Namespace
from typing import Tuple, List

import numpy as np
from utils.exceptions import UnknownNameCustomEnumException
from utils.types import DatasetName, ClientsDataStatistics

from .agnews import get_agnews_data


def get_dataset(args: Namespace) -> Tuple[List[np.ndarray], List[np.ndarray], ClientsDataStatistics]:
    config = {
        'dataset_name': args.dataset,
        'num_clients': args.num_clients,
        'num_classes': args.num_classes,
        'alpha': args.alpha,
        'niid': args.niid,
        'balance': args.balance,
        'partition': args.partition,
        'batch_size': args.batch_size,
        'train_size': 0.8,
        'ref': args.ref,
        'class_per_client': args.class_per_client
    }
    config = Namespace(**config)

    match args.dataset:
        case DatasetName.agnews:
            return get_agnews_data(config)
        case _:
            raise UnknownNameCustomEnumException(args.dataset, DatasetName)
