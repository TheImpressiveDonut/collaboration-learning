from argparse import Namespace
from typing import List, Tuple

import numpy as np
import tiktoken
import torchtext
from utils.data import clients_split, train_test_ref_split
from utils.folders import get_raw_path, check_if_config_exist
from utils.memory import save_dataset, load_dataset
from utils.types import ClientsDataStatistics


def get_agnews_data(args: Namespace) -> Tuple[List[np.ndarray], List[np.ndarray], ClientsDataStatistics]:
    if check_if_config_exist(args):
        (train, eval, _), stats = load_dataset(args)
        return train, eval, stats

    raw_path = get_raw_path(args.dataset_name)
    trainset, testset = torchtext.datasets.AG_NEWS(root=raw_path)

    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)
    dataset_text = np.array(dataset_text)
    dataset_label = np.array(dataset_label)

    clients_data, statistic = clients_split((dataset_text, dataset_label - 1), args)
    train_data, test_data, _ = train_test_ref_split(clients_data, args)

    tokenizer = tiktoken.get_encoding('gpt2')
    tokenized_train_data = []
    tokenized_test_data = []
    for i in range(len(train_data)):
        train_combined_texts = ' '.join(train_data[i][0])
        test_combined_texts = ' '.join(test_data[i][0])

        raw_tokenized_train = tokenizer.encode_ordinary(train_combined_texts)
        raw_tokenized_eval = tokenizer.encode_ordinary(test_combined_texts)
        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
        test_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)
        tokenized_train_data.append(train_tokenized)
        tokenized_test_data.append(test_tokenized)

    save_dataset(tokenized_train_data, tokenized_test_data, [], statistic, args)

    del raw_tokenized_train, raw_tokenized_eval, train_tokenized, test_tokenized
    return tokenized_train_data, tokenized_test_data, statistic
