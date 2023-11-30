from argparse import Namespace
from typing import List, Dict, Tuple

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score
from torch import Tensor, nn

from src.models.netcnn import NetCNN
from src.models.netfnn import NetFNN
from src.models.resnet import ResNet, resnet20
from src.utils.exceptions import UnknownNameCustomEnumException
from src.utils.types import DatasetName, ModelName, ModeName, SettingName


class CosineSimilarity(nn.Module):
    def __init__(self, regularized: bool = True) -> None:
        super(CosineSimilarity, self).__init__()
        self.regularized = regularized

    def forward(self, input: Tensor, target: Tensor) -> float:
        sim = nn.CosineSimilarity(dim=1)(input, target)
        if self.regularized:
            ent = -torch.sum(target * torch.log(target + 1e-8), dim=1) + 1.0
            # prevent entropy being too small
            sim = torch.div(sim, ent)
        return torch.mean(sim).item()


class TrueLabelSimilarity(nn.Module):
    def __init__(self, id: int, labels: List[Tensor], num_clients: int, device: torch.device) -> None:
        super(TrueLabelSimilarity, self).__init__()
        self.id = id
        self.labels = labels
        self.lengs = [len(self.labels[id]) for id in range(num_clients)]
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> float:
        true_lbs = self.labels[self.id].to(self.device)
        idx_start = int(0 + sum(self.lengs[:self.id]))
        idx_end = int(sum(self.lengs[:self.id + 1]))
        pred_lbs = torch.argmax(input, dim=1)[idx_start:idx_end]
        return accuracy_score(pred_lbs, true_lbs)


def __select_model(args: Namespace, model_name: ModelName) -> [ResNet | NetCNN | EfficientNet]:
    match args.dataset_name:
        case DatasetName.cifar10 | DatasetName.cifar100:
            match model_name:
                case ModelName.resnet:
                    return resnet20(num_classes=args.num_classes)
                case ModelName.cnn:
                    return NetCNN(in_features=args.num_channels, num_classes=args.num_classes, dim=1600)
                case ModelName.fnn:
                    return NetFNN(input_dim=3 * 32 * 32, mid_dim=100, num_classes=args.num_classes)
                case _:
                    raise NotImplementedError
        case DatasetName.fed_isic:
            match args.model_name:
                case ModelName.efficient_net:
                    return EfficientNet.from_pretrained(args.eff_net_arch_name, num_classes=args.num_classes)
                case ModelName.cnn:
                    return NetCNN(in_features=args.num_channels, num_classes=args.num_classes, dim=10816)
                case ModelName.fnn:
                    return NetFNN(input_dim=3 * 64 * 64, mid_dim=100, num_classes=8)
                case _:
                    raise NotImplementedError
        case _:
            raise UnknownNameCustomEnumException(args.dataset_name, DatasetName)

def select_client_model_mode(args: Namespace) -> Dict[int, Tuple[ResNet | NetCNN | EfficientNet, ModeName]]:
    clients = {}
    match args.setting:
        case SettingName.normal:
            match args.dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    for i in range(args.num_clients):
                        clients[i] = (__select_model(args, ModelName.resnet), ModeName.normal)
                case DatasetName.fed_isic:
                    for i in range(args.num_clients):
                        clients[i] = (__select_model(args, ModelName.efficient_net), ModeName.normal)
                case _:
                    raise UnknownNameCustomEnumException(args.dataset_name, DatasetName)

        case SettingName.two_sets:
            middle = args.num_clients // 2
            match args.dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    for i in range(middle):
                        clients[i] = (__select_model(args, ModelName.resnet), ModeName.normal)
                    for i in range(middle, args.num_clients):
                        clients[i] = (__select_model(args, ModelName.fnn), ModeName.normal)
                case DatasetName.fed_isic:
                    for i in range(middle):
                        clients[i] = (__select_model(args, ModelName.efficient_net), ModeName.normal)
                    for i in range(middle, args.num_clients):
                        clients[i] = (__select_model(args, ModelName.fnn), ModeName.normal)
                case _:
                    raise UnknownNameCustomEnumException(args.dataset_name, DatasetName)

        case SettingName.evil:
            client_ids = np.arange(args.num_clients)
            match args.dataset_name:
                case DatasetName.cifar10 | DatasetName.cifar100:
                    evil_idx = np.array([2, 9])
                    print('evil worker:', evil_idx)
                    normal_idx = [id for id in client_ids if id not in evil_idx]
                    for i in evil_idx:
                        clients[i] = (__select_model(args, ModelName.resnet), ModeName.flipped)
                    for i in normal_idx:
                        clients[i] = (__select_model(args, ModelName.resnet), ModeName.normal)
                case DatasetName.fed_isic:
                    evil_idx = np.array([1, 2])
                    print('evil worker:', evil_idx)
                    normal_idx = [id for id in client_ids if id not in evil_idx]
                    for i in evil_idx:
                        clients[i] = (__select_model(args, ModelName.efficient_net), ModeName.flipped)
                    for i in normal_idx:
                        clients[i] = (__select_model(args, ModelName.efficient_net), ModeName.normal)
                case _:
                    raise UnknownNameCustomEnumException(args.dataset_name, DatasetName)
        case _:
            raise UnknownNameCustomEnumException(args.setting, SettingName)

    return clients