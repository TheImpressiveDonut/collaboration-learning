from typing import Dict

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from client.client import Client
from utils.exceptions import UnknownNameCustomEnumException
from utils.types import MetricName


class Trainer(object):

    def __init__(self,
                 clients: Dict[int, Client],
                 ref_loader: DataLoader,
                 clients_sample_size: np.ndarray,
                 pretraining_rounds: int,
                 num_local_epochs: int,
                 metric: MetricName,
                 ) -> None:
        self.clients = clients
        self.ref_loader = ref_loader
        self.clients_sample_size = clients_sample_size
        self.pretraining_rounds = pretraining_rounds
        self.num_local_epochs = num_local_epochs
        self.metric_name = metric
        match metric:
            case MetricName.acc:
                self.metric = accuracy_score
                # replaced this sum(ref_pred == ref_y) / len(ref_y) by accuracy score
            case MetricName.bacc:
                self.metric = balanced_accuracy_score
            case _:
                raise UnknownNameCustomEnumException(metric, MetricName)

        self.soft_decisions = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.ref_accuracies = []
        self.trust_weights_round = {}

    def train(self, global_epochs: int):
        for current_global_epoch in tqdm(range(global_epochs), position=0, leave=False, desc='global epochs'):
            self.trust_weights_round[current_global_epoch] = []
            self.__global_epoch(current_global_epoch)

            print('round %i finished' % current_global_epoch)
            print('local accuracy after round', str(current_global_epoch), ':',
                  np.mean(self.test_accuracies[current_global_epoch]))
            print('local accuracy:', self.test_accuracies[current_global_epoch])
            print('global accuracy after round', str(current_global_epoch), ':',
                  np.mean(self.ref_accuracies[current_global_epoch]))
            print('global acc:', self.ref_accuracies[current_global_epoch])

            if len(self.trust_weights_round[current_global_epoch]) > 0:
                self.trust_weights_round[current_global_epoch] = torch.stack(
                    self.trust_weights_round[current_global_epoch])

        return self.train_accuracies, self.test_accuracies, self.ref_accuracies, self.soft_decisions

    def __global_epoch(self, current_global_epoch: int):

        # @todo some of parallel or bigger batch_size
        for idx, client in tqdm(self.clients.items(), position=1, leave=False, desc='clients'):
            if current_global_epoch < self.pretraining_rounds:
                soft_decision_target = None
            else:
                soft_decision_target, trust_weight = client.calculate_soft_decision_target(
                    self.soft_decisions[current_global_epoch - 1], current_global_epoch)
                self.trust_weights_round[current_global_epoch].append(trust_weight)

            for current_local_epoch in tqdm(range(self.num_local_epochs), position=2, leave=False, desc='local epoch'):  # @todo change local epochs over time
                self.__local_epoch(client, soft_decision_target)

            self.train_accuracies.append(client.test(self.metric_name, mode='train'))
            soft_decision = client.infer_on_refdata(self.ref_loader)
            self.soft_decisions.append(soft_decision)

            self.test_accuracies.append(client.test(self.metric_name, mode='test'))

            ref_pred = torch.argmax(soft_decision, dim=1)
            self.ref_accuracies.append(self.metric(self.ref_loader, ref_pred))

    def __local_epoch(self, client: Client, soft_decision_target: Tensor):
        client.train(self.ref_loader, soft_decision_target)
