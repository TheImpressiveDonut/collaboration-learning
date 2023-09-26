import os
import random
from typing import Callable
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from utils.types import Data


class PDLDataSet(torch.utils.data.Dataset):
    def __init__(self, data: Data, mode: str = 'normal', num_class: int = 10) -> None:
        self.X = data[0]
        if mode == 'normal':
            self.y = data[1]
        elif mode == 'randomized':
            self.y = random.sample(list(data[1]), len(data[1]))
        elif mode == 'flipped':
            label_shift = random.randint(1, num_class - 1)
            self.y = list((data[1] + label_shift) % num_class)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = self.X[idx, :, :, :]
        label = self.y[idx]
        return img, label

# @todo maybe juste rename SharedDara
class SharedData(torch.utils.data.Dataset):
    def __init__(self, X: Tensor, Y: Tensor) -> None:
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.size()[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        img = self.X[idx, :, :, :]
        label = self.Y[idx]
        return img, label, idx


# @todo better way to do that
def dl_to_sampler(dl) -> Callable[[], Tensor]:
    dl_iter = iter(dl)

    def sample() -> Tensor:
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)

    return sample





# @todo to remove
def get_ref_data(curr_path: str) -> Tuple[Tensor, Tensor]:
    ref_x = []
    ref_y = []
    # ref_client = []
    for dir in os.listdir(curr_path):
        tmp_path = os.path.join(curr_path, dir)
        with open(tmp_path, 'rb') as f:
            tmp = np.load(f, allow_pickle=True)['data'].tolist()
        ref_x.extend(tmp['x'])
        ref_y.extend(tmp['y'])
        # ref_client.extend(dir[:-4]*len(ref_y))
    ref_X = torch.tensor(np.array(ref_x))
    ref_y = torch.tensor(ref_y)
    return ref_X, ref_y

# @todo to remove
def get_ref_labels_by_client(curr_path: str) -> Dict[]:
    ref_labels_by_clients = {}
    for dir in os.listdir(curr_path):
        tmp_path = os.path.join(curr_path, dir)
        client_id = int(dir[:-4])
        with open(tmp_path, 'rb') as f:
            tmp = np.load(f, allow_pickle=True)['data'].tolist()
        ref_labels_by_clients[client_id] = tmp['y']
    return ref_labels_by_clients


def save_results(trust_weight_dict, test_accuracy_dict, ref_accuracy_dict, res_path, args) -> None:
    ref_acc_path = os.path.join(res_path, 'global_accuracy_' + str(
        args.dataset_name) + '_' + args.consensus_mode + '_' + args.sim_measure + '_' + args.trust_update + '_' + 'lam_' + str(
        args.lambda_) + '_' + str(args.experiment_no) + '_' + 'local_epoch_' + str(
        args.num_local_epochs) + '_' + args.setting + '.pt')

    test_acc_path = os.path.join(res_path, 'local_accuracy_' + str(
        args.dataset_name) + '_' + args.consensus_mode + '_' + args.sim_measure + '_' + args.trust_update + '_' + 'lam_' + str(
        args.lambda_) + '_' + str(args.experiment_no) + '_' + 'local_epoch_' + str(
        args.num_local_epochs) + '_' + args.setting + '.pt')

    trust_weight_path = os.path.join(res_path, 'trust_weight_dict_' + str(
        args.dataset_name) + '_' + args.consensus_mode + '_' + args.sim_measure + '_' + args.trust_update + '_' + 'lam_' + str(
        args.lambda_) + '_' + str(args.experiment_no) + '_' + 'local_epoch_' + str(
        args.num_local_epochs) + '_' + args.setting + '.pt')

    torch.save(trust_weight_dict, trust_weight_path)
    torch.save(test_accuracy_dict, test_acc_path)
    torch.save(ref_accuracy_dict, ref_acc_path)
