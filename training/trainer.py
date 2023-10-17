from typing import Dict, List

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

    def train(self, global_epochs: int, local_validation: bool = False, find_collaborators: bool = False):
        global_epochs = tqdm(range(global_epochs), position=0, leave=False, desc='global epochs')
        prev_test_accuracy = 0
        prev_ref_accuracy = 0

        for current_global_epoch in global_epochs:
            self.trust_weights_round[current_global_epoch] = []
            self.soft_decisions.append([])
            self.train_accuracies.append([])
            self.test_accuracies.append([])
            self.ref_accuracies.append([])

            self.__global_epoch(current_global_epoch)

            if len(self.trust_weights_round[current_global_epoch]) > 0:
                self.trust_weights_round[current_global_epoch] = torch.stack(
                    self.trust_weights_round[current_global_epoch])

            test_accuracy = np.mean(self.test_accuracies[current_global_epoch])
            ref_accuracy = np.mean(self.ref_accuracies[current_global_epoch])
            if current_global_epoch == 0:
                prev_test_accuracy = test_accuracy
                prev_ref_accuracy = ref_accuracy
            global_epochs.set_description(
                f'global epochs (local accuracy: {test_accuracy:.5f}[{test_accuracy - prev_test_accuracy:+.5f}]'
                f' | global accuracy: {ref_accuracy:.5f}[{ref_accuracy - prev_ref_accuracy:+.5f}])' if not find_collaborators else ''
            )

            prev_test_accuracy = test_accuracy
            prev_ref_accuracy = ref_accuracy

        return self.trust_weights_round, self.test_accuracies, self.ref_accuracies

    def __global_epoch(self, current_global_epoch: int, find_collaborators: bool = False):

        # @todo some parallel
        for idx, client in tqdm(self.clients.items(), position=1, leave=False, desc='clients'):
            if current_global_epoch < self.pretraining_rounds or not find_collaborators:
                soft_decision_target = None
            else:
                soft_decision_target, trust_weight = client.calculate_soft_decision_target(
                    current_global_epoch, self.soft_decisions[current_global_epoch - 1])
                self.trust_weights_round[current_global_epoch].append(trust_weight)

            for current_local_epoch in tqdm(range(self.num_local_epochs), position=2, leave=False,
                                            desc='local epoch'):  # @todo change local epochs over time
                self.__local_epoch(client, soft_decision_target)

            self.train_accuracies[current_global_epoch].append(client.test(self.metric_name, mode='train'))
            soft_decision, ref_y = client.infer_on_refdata(self.ref_loader)
            self.soft_decisions[current_global_epoch].append(soft_decision)

            self.test_accuracies[current_global_epoch].append(client.test(self.metric_name, mode='test'))

            ref_pred = torch.argmax(soft_decision, dim=1)
            self.ref_accuracies[current_global_epoch].append(self.metric(ref_y, ref_pred))

    def __local_epoch(self, client: Client, soft_decision_target: Tensor):
        client.train(self.ref_loader, soft_decision_target)
