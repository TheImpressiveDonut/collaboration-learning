# data-splits and source from https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

from utils.data import train_test_ref_split
from utils.folders import get_raw_path
from utils.types import Config, ClientsData, ClientsDataStatistics


def fed_isic(config: Config) -> Tuple[
    ClientsData, ClientsData, ClientsData, ClientsDataStatistics
]:
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    raw_path = get_raw_path(config.dataset_name, create_dir=False)
    image_dir = f'{raw_path}/ISIC_2019_Training_Input_preprocessed'
    data = pd.read_csv(f'{raw_path}/ISIC_2019_Training_Metadata_FL.csv', delimiter=',')
    # image, age_approx, anatom_site_general, sex, datasets
    labels = pd.read_csv(f'{raw_path}/fed-isic-2019/ISIC_2019_Training_GroundTruth.csv', delimiter=',')
    # image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK

    labels_new = {'image': labels['image'],
                  'label': np.where(labels.iloc[:, 1:] != 0)[1]}
    labels_new = pd.DataFrame(labels_new)
    big_table = pd.merge(data, labels_new, on='image', how="inner")

    num_rows = big_table.shape[0]
    client_names = np.unique(big_table['datasets'])
    clients_data = []
    for j in range(config.num_clients):
        clients_data.append((np.array([]), np.array([])))

    for i in range(num_rows):
        for j in range(config.num_clients):
            if big_table['datasets'][i] == client_names[j]:
                image_name = os.path.join(image_dir, big_table['image'][i] + '.jpg')
                img = transform(Image.open(image_name))
                clients_data[j] = np.stack((clients_data[j][0], img.numpy()), axis=0), np.append(clients_data[j][1],
                                                                                                 big_table['label'][i])
    for j in range(config.num_clients):
        print(clients_data[j][0].shape)

    statistics = []
    for client in range(config.num_clients):
        statistics.append([])
        for val, count in np.unique(clients_data[client][1], return_counts=True):
            statistics[client].append((val, count))

    train_data, test_data, ref_data = train_test_ref_split(clients_data, config, mode='by-number')
    return train_data, test_data, ref_data, statistics
