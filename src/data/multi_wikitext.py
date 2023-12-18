import os
from argparse import Namespace
from typing import List, Tuple

import datasets
import numpy as np
import tiktoken
from datasets import load_from_disk
from utils.data import clients_split, train_test_ref_split
from utils.folders import check_if_config_exist
from utils.folders import get_raw_path
from utils.memory import load_dataset, save_dataset
from utils.types import ClientsDataStatistics


def get_multi_wikitext_data(args: Namespace) -> Tuple[List[np.ndarray], List[np.ndarray], ClientsDataStatistics]:
    if check_if_config_exist(args):
        (train, eval, _), stats = load_dataset(args)
        return train, eval, stats

    raw_path = get_raw_path(args)

    # Get multi-lingual wikipedia data
    print('This data downloading process might take a while... be patient.')

    dataset_text = []
    dataset_label = []
    i = 0
    for dataset_idx in ['20220301.de', '20220301.it']:
                        #'20220301.fr', '20220301.it']:
        dataset_path = os.path.join(raw_path, dataset_idx)
        if os.path.isdir(dataset_path):
            print('loading from disk: ', dataset_idx)
            data_one_lang = load_from_disk(dataset_path)
        else:
            data_one_lang = datasets.load_dataset('wikipedia', dataset_idx)
            data_one_lang.save_to_disk(dataset_path)
        dataset_text.extend(data_one_lang['train']['text'])
        l = len(data_one_lang['train']['text'])
        del data_one_lang
        dataset_label.extend([i] * l)
        i = i + 1

    # Tokenize the data
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset_text = np.array(dataset_text)
    dataset_label = np.array(dataset_label)
    print('sample 10\% of the data')
    sampled_indices = np.random.choice(np.arange(len(dataset_label)), size=int(0.1 * len(dataset_label)),
                                       replace=False).astype(int)

    clients_data, statistic = clients_split((dataset_text[sampled_indices], dataset_label[sampled_indices]), args)
    del dataset_text, dataset_label

    train_data, test_data, _ = train_test_ref_split(clients_data, args)
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
    return tokenized_train_data, tokenized_test_data, statistic
