from argparse import Namespace
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.exceptions import UnknownPartitionArgumentException, UnknownSplitDataModeException
from src.utils.types import ClientsData, Data, ClientsDataStatistics


def __get_pat_partition(dataset_label: np.ndarray, args: Namespace) -> List[np.ndarray]:
    data_idx_map = [np.array([], dtype=int) for _ in range(args.num_clients)]
    least_samples = args.batch_size / (1 - args.train_size)  # remove batch size
    idx_for_each_class = []
    for i in range(args.num_classes):
        idx_for_each_class.append(np.argwhere(dataset_label == i).flatten())
    class_num_per_client = np.full(args.num_clients, args.class_per_client, dtype=int)

    for i in range(args.num_classes):
        selected_clients = np.argwhere(class_num_per_client > 0).flatten()
        selected_clients = selected_clients[:int(args.num_clients / args.num_classes * args.class_per_client)]

        num_all_samples = idx_for_each_class[i].shape[0]
        num_selected_clients = selected_clients.shape[0]
        num_sample_per_clients = int(num_all_samples / num_selected_clients)

        if args.balance:
            idxs = np.array_split(idx_for_each_class[i], num_sample_per_clients)
        else:
            num_samples = np.random.randint(
                int(max(num_sample_per_clients / 10, least_samples / args.num_classes)),
                num_sample_per_clients, num_selected_clients, dtype=int)
            idxs = np.split(idx_for_each_class[i], np.cumsum(num_samples[:-1]))

        for client in selected_clients:
            data_idx_map[client] = np.concatenate((data_idx_map[client], idxs[client]))
            class_num_per_client[client] -= 1

    return data_idx_map


def __get_dir_partition(dataset_label: np.ndarray, args: Namespace) -> List[np.ndarray]:
    data_idx_map = [np.array([], dtype=int) for _ in range(args.num_clients)]
    least_samples = args.batch_size / (1 - args.train_size)
    # inspired from https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    min_size = 0
    K = args.num_classes
    N = dataset_label.shape[0]

    while min_size < least_samples:
        idx_batch = [[] for _ in range(args.num_clients)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.num_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / args.num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(args.num_clients):
        data_idx_map[j] = np.array(idx_batch[j], dtype=int)

    return data_idx_map


def clients_split(data: Data, args: Namespace) -> Tuple[ClientsData, ClientsDataStatistics]:
    clients_data, statistic = [], []
    dataset_image, dataset_label = data

    match args.partition:
        case 'pat':
            data_idx_map = __get_pat_partition(dataset_label, args)
        case 'dir':
            data_idx_map = __get_dir_partition(dataset_label, args)
        case _:
            raise UnknownPartitionArgumentException(args.partition)

    # assign data
    for client in range(args.num_clients):
        idxs = data_idx_map[client]
        clients_data.append((dataset_image[idxs], dataset_label[idxs]))

        statistic.append([])
        vals, counts = np.unique(clients_data[client][1], return_counts=True)
        for val, count in zip(vals, counts):
            statistic[client].append((int(val), int(count)))

    for client in range(args.num_clients):
        print(f'Client {client}\t Size of data: {clients_data[client][0].shape[0]}\t Labels: ',
              np.unique(clients_data[client][1]))
        print(f'\t\t Samples of labels: ', [i for i in statistic[client]])
        print('-' * 50)

    return clients_data, statistic


def __get_num_samples(data: ClientsData) -> int:
    return sum(map(lambda x: x[1].shape[0], data))


def train_test_ref_split(clients_data: ClientsData, args: Namespace, mode: str = 'by-ratio'
                         ) -> Tuple[ClientsData, ClientsData, ClientsData]:
    train_data, test_data, ref_data = [], [], []

    # For each client
    for i in range(len(clients_data)):
        X_train, X_test, y_train, y_test = train_test_split(
            clients_data[i][0], clients_data[i][1], train_size=args.train_size, shuffle=True)

        if args.ref:
            match mode:
                case 'by-ratio':
                    # 80% test_folder, 20% ref
                    X_test, X_ref, y_test, y_ref = train_test_split(
                        X_test, y_test, train_size=0.8, shuffle=True)
                case 'by-number':
                    # take 50 random elements of test_folder for ref
                    sampled_idx = np.random.choice(X_test.shape[0], min(X_test.shape[0], 50), replace=False).astype(int)
                    X_ref, y_ref = X_test[sampled_idx, :, :, :], y_test[sampled_idx]
                    X_test, y_test = X_test[~sampled_idx, :, :, :], y_test[~sampled_idx]
                case _:
                    raise UnknownSplitDataModeException(mode)

            ref_data.append((X_ref, y_ref))

        train_data.append((X_train, y_train))
        test_data.append((X_test, y_test))

    train_num, test_num, ref_num = __get_num_samples(train_data), __get_num_samples(test_data), __get_num_samples(
        ref_data)
    print('Total number of samples: ', train_num + test_num + ref_num)
    print('The number of train samples: ', train_num)
    print('The number of test_folder samples: ', test_num)
    if args.ref:
        print('The number of ref samples:', ref_num)

    return train_data, test_data, ref_data
