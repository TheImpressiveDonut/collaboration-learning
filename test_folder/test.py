import torch
from torch import Tensor

from test_folder.setup import CosineSimilarity
from test_folder.utils import timeit, print_size


@timeit
def normal(soft_decision_list: Tensor) -> Tensor:
    sim = CosineSimilarity(True)
    sd_eigen = soft_decision_list[:, 1].unsqueeze(1)
    sim_list = []
    for j in range(soft_decision_list.size(dim=1)):
        sd_other = soft_decision_list[:, j].unsqueeze(1)
        sim_ = sim(sd_other, sd_eigen)
        sim_list.append(sim_)
    sim_list = torch.tensor(sim_list)

    trust_weights = (sim_list / torch.sum(sim_list)).unsqueeze(1)
    return trust_weights


if __name__ == '__main__':
    torch.manual_seed(1)
    init = torch.abs(torch.randn((100000, 10)))
    print_size(init, 'init')
    # res1 = normal(init)
    # print_size(res1, 'res1')
    # print(res1)
    weight = torch.randn((10, 1))
    print_size(weight, 'weight')
    weighted_soft_decisions = torch.permute(weight, (1, 0)) * init
    print_size(weighted_soft_decisions, 'soft')
    target = torch.sum(weighted_soft_decisions, dim=1)
    print_size(target, 'target')
    target = torch.nn.functional.normalize(target, p=1.0, dim=0)
    print_size(target, 'target')
