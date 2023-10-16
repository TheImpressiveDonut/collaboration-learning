import torch
from torch import Tensor, nn

from test_folder.utils import print_size


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
