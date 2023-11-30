import torch
import torch.nn as nn
from torch import Tensor


class NetFNN(nn.Module):
    def __init__(self, input_dim: int = 3 * 32 * 32, mid_dim: int = 100, num_classes: int = 10) -> None:
        super(NetFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        #  self.fc = nn.Linear(mid_dim, num_classes)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x
