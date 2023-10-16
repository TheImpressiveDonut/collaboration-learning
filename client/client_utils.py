from typing import List

import torch
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score
from torch import Tensor, nn

from models.netcnn import NetCNN
from models.netfnn import NetFNN
from models.resnet import ResNet, resnet20
from utils.exceptions import UnknownNameCustomEnumException
from utils.types import DatasetName, ModelName


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


def select_client_model(
        dataset_name: DatasetName,
        model_name: ModelName,
        num_classes: int,
        num_channels: int,
        eff_net_arch_name: str,
) -> [ResNet | NetCNN | EfficientNet]:
    # @todo check arbitrary/magic numbers
    match dataset_name:
        case DatasetName.cifar10 | DatasetName.cifar100:
            match model_name:
                case ModelName.resnet:
                    return resnet20(num_classes=num_classes)
                case ModelName.cnn:
                    return NetCNN(in_features=num_channels, num_classes=num_classes, dim=1600)
                case ModelName.fnn:
                    return NetFNN(input_dim=3 * 32 * 32, mid_dim=100, num_classes=num_classes)
                case _:
                    raise NotImplementedError
        case DatasetName.fed_isic:
            match model_name:
                case ModelName.efficient_net:
                    return EfficientNet.from_pretrained(eff_net_arch_name, num_classes=num_classes)
                case ModelName.cnn:
                    return NetCNN(in_features=num_channels, num_classes=num_classes, dim=10816)
                case ModelName.fnn:
                    return NetFNN(input_dim=3 * 64 * 64, mid_dim=100, num_classes=8)
                case _:
                    raise NotImplementedError
        case _:
            raise UnknownNameCustomEnumException(dataset_name, DatasetName)
