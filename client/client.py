import os
from argparse import Namespace
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader

from client_utils import select_client_model
from models.data_utils import PDLDataSet, get_ref_labels_by_client, dl_to_sampler
from utils.types import Data


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return torch.mean(-torch.sum(target * torch.log(input + 1e-8), 1))


class CosineSimilarity(nn.Module):
    def __init__(self, mode: str = 'regularized') -> None:
        super(CosineSimilarity, self).__init__()
        self.mode = mode

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        sim = nn.CosineSimilarity(dim=1, eps=1e-8)(input, target)
        # simi = sim(input, target) @Todo Remove once correctness is verified
        if self.mode == 'regularized':
            ent = -torch.sum(target * torch.log(target + 1e-8), dim=1) + 1.0
            # prevent entropy being too small
            reg_sim = torch.div(sim, ent)
        elif self.mode == 'normal':
            reg_sim = sim
        else:
            raise NotImplementedError
        return torch.mean(reg_sim)


class TrueLabelSimilarity(nn.Module):
    def __init__(self, id: int, labels: Dict[int,], num_clients: int, device: torch.device) -> None:
        super(TrueLabelSimilarity, self).__init__()
        self.id = id
        self.labels = labels
        self.lengs = [len(self.labels[id]) for id in range(num_clients)]
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # @Todo what's the point of doing this every forward ? single usage, confirm !
        true_lbs = torch.Tensor(self.labels[self.id]).to(self.device)
        idx_start = int(0 + sum(self.lengs[:self.id]))
        idx_end = int(sum(self.lengs[:self.id + 1]))
        pred_lbs = torch.argmax(input, dim=1)[idx_start:idx_end]
        return torch.sum(true_lbs == pred_lbs) / len(true_lbs)


class Client(object):
    def __init__(
            self,
            worker_index: int,
            train_data: Data,
            test_data: Data,
            args: Namespace,
            model_name: str = 'resnet',
            mode: str = 'normal'

    ) -> None:
        self.model = select_client_model(args.dataset_name, model_name)

        self.device = args.device
        self.consensus = args.consensus_mode
        self.id = worker_index
        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.num_classes = args.num_classes
        self.train_batch_size: int = args.train_batch_size
        self.test_batch_size: int = args.test_batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.trust = args.trust_update
        self.data_path = os.path.join(args.dataset_path, args.dataset_name)
        self.ref_labels = get_ref_labels_by_client(os.path.join(self.data_path, 'ref'))

        if args.sim_measure == 'cosine':
            self.sim = CosineSimilarity(args.cmode)
            self.task_difficulty_switch = False
        elif args.sim_measure == 'true_label':
            self.sim = TrueLabelSimilarity(self.id, self.ref_labels, args.num_clients, self.device)
            self.task_difficulty_switch = False

        self.train_dataset = PDLDataSet(train_data, mode=mode, num_class=self.num_classes)
        self.test_dataset = PDLDataSet(test_data, mode='normal')

        # train_sampler = DistributedSampler(datasets=self.train_dataset, shuffle=True) # @todo check this

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                       pin_memory=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        self.lambda_ = args.lambda_
        self.trust_weights: Tensor = []
        self.num_clients: int = args.num_clients
        self.period: int = args.trust_update_frequency
        self.prer: int = args.pretraining_rounds

    def train(self, epoch: int, extra_data: DataLoader,
              soft_decision_target: Tensor = None) -> None:  # @todo epoch useless ?
        ref_sample = dl_to_sampler(extra_data)
        self.model.to(self.device)
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            with torch.no_grad():
                extra_data_sampled, _, ref_idx = ref_sample()
                extra_data_sampled = extra_data_sampled.to(self.device)
            soft_decision = F.softmax(self.model(extra_data_sampled), dim=1)
            local_loss = torch.nn.CrossEntropyLoss()(output, target)
            if soft_decision_target is None:
                soft_decision_loss = 0
            else:
                soft_decision_target = soft_decision_target.to(self.device)
                soft_decision_loss = cross_entropy(soft_decision, soft_decision_target[ref_idx, :])
            loss = local_loss + self.lambda_ * soft_decision_loss
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def infer_on_refdata(self, ref_loader: DataLoader) -> Tensor:
        soft_decision_list = []
        self.model.eval()
        with torch.no_grad():  # @todo already an annotation
            for batch_idx, (data, _, _) in enumerate(ref_loader):
                data = data.to(self.device)
                soft_decision = F.softmax(self.model(data), dim=1)
                soft_decision_list.append(soft_decision)
        soft_decisions = torch.cat(soft_decision_list, dim=0)
        return soft_decisions.detach().cpu()

    @torch.no_grad()
    def soft_assignment_to_hard_assignment(self, soft_decisions: Tensor) -> Tensor:
        a = torch.argmax(soft_decisions, 1)
        hard_decisions = torch.zeros_like(soft_decisions)
        for i in range(hard_decisions.shape[0]):
            hard_decisions[i, a[i]] = 1
        return hard_decisions

    @torch.no_grad()
    def calculate_trust_weight_mat(self, soft_decision_list: List[Tensor], train_acc_list: List[Tensor],
                                   clients_sample_size: int) -> float:
        sd_eigen = soft_decision_list[self.id]
        sim_list = []
        for j in range(len(soft_decision_list)):
            sd_other = soft_decision_list[j]
            sim_ = self.sim(sd_other.to(self.device), sd_eigen.to(self.device))
            sim_list.append(sim_)
        sim_list = torch.tensor(sim_list)
        if not train_acc_list:
            pass
        else:
            if self.task_difficulty_switch:
                local_loss_list = (1 - torch.Tensor(train_acc_list) + 1e-5) / clients_sample_size
                local_loss_list -= torch.min(local_loss_list)
                local_loss_list /= torch.max(local_loss_list)
                local_loss_list += 1
                sim_list = sim_list / local_loss_list

        trust_weights = sim_list / torch.sum(sim_list)
        self.trust_weights = trust_weights
        return trust_weights

    @torch.no_grad()
    def calculate_soft_decision_target(self, soft_decision_list: List[Tensor], train_acc_list: List[Tensor],
                                       clients_sample_size: int, global_round: int
                                       ) -> Tuple[Tensor, Tensor]:
        if self.consensus == 'soft_assignment':
            if self.trust == 'static':
                if not self.trust_weights:
                    self.trust_weights = self.calculate_trust_weight_mat(soft_decision_list, train_acc_list,
                                                                         clients_sample_size)
            elif self.trust == 'dynamic':
                if (global_round - self.prer) % self.period == 0:
                    self.trust_weights = self.calculate_trust_weight_mat(soft_decision_list, train_acc_list,
                                                                         clients_sample_size)
            elif self.trust == 'naive':
                self.trust_weights = torch.ones(self.num_clients, 1)
                self.trust_weights = self.trust_weights / self.num_clients
            else:
                raise NotImplementedError

            weighted_soft_decisions = [self.trust_weights[i] * soft_decision_list[i] for i in range(self.num_clients)]
            target = sum(weighted_soft_decisions)
            target = torch.nn.functional.normalize(target, p=1.0, dim=1)
            return target, self.trust_weights

        elif self.consensus == 'majority_voting':
            hard_labels = [self.soft_assignment_to_hard_assignment(soft_decision_list[i]) for i in
                           range(self.num_clients)]
            target = sum(hard_labels)
            target = self.soft_assignment_to_hard_assignment(target)
            self.trust_weights = torch.ones(self.num_clients, 1) / self.num_clients
            return target, self.trust_weights

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def test(self, metric: str = 'acc', mode: str = 'test') -> float:
        if mode == 'train':
            loader = self.train_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise NotImplementedError

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():  # @todo already annotation no grad
            if metric == 'acc':
                preds = []
                for batch_idx, (data, target) in enumerate(loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    pred_test = torch.sum(torch.argmax(self.model(data), dim=1) == target)
                    preds.append(pred_test.detach().cpu())
                return np.sum(preds) / len(loader.dataset)
            elif metric == 'bacc':
                pred_test_list = []
                target_list = []
                for batch_idx, (data, target) in enumerate(loader):
                    data = data.to(self.device)
                    target_list.extend(target)
                    pred_test = torch.argmax(self.model(data), dim=1).detach().cpu()
                    pred_test_list.append(pred_test)
                pred_test = torch.cat(pred_test_list, 0)
                bacc = balanced_accuracy_score(target_list, pred_test)
                return bacc

    def get_train_set_size(self) -> int:
        return self.train_loader.__len__()
