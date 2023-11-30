from argparse import Namespace
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader

from src.client.client_utils import __select_model, CosineSimilarity, TrueLabelSimilarity
from src.models.models_utils import PDLDataSet, dl_to_sampler
from src.models.netcnn import NetCNN
from src.models.resnet import ResNet
from src.utils.exceptions import UnknownNameCustomEnumException
from src.utils.types import Data, ModelName, ClientsData, TrustName, ConsensusName, MetricName, ModeName
from src.utils.types import DatasetName, SimMeasureName


class Client(object):
    def __init__(self,
                 id: int, model: [ResNet | NetCNN | EfficientNet], mode: ModeName, device: torch.device,
                 train_data: Data, test_data: Data, ref_data: ClientsData,
                 args: Namespace
                 ) -> None:
        self.id = id
        self.device = device
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        self.consensus = args.consensus_mode
        self.trust = args.trust_update

        self.ref_labels = list(map(lambda x: torch.from_numpy(x[1]), ref_data))

        self.lambda_ = args.lambda_
        self.num_clients = args.num_clients
        self.trust_update_frequency = args.trust_update_frequency
        self.pretraining_rounds = args.pretraining_rounds
        self.trust_weights = None  # Tensor of size (num_clients, 1)

        match args.sim_measure:
            case SimMeasureName.cosine:
                self.sim = CosineSimilarity(args.cmode)
            case SimMeasureName.true_label:
                self.sim = TrueLabelSimilarity(self.id, self.ref_labels, args.num_clients, self.device)
            case _:
                raise UnknownNameCustomEnumException(args.sim_measure, SimMeasureName)

        self.train_dataset = PDLDataSet(train_data, mode=mode, num_class=args.num_classes)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                       pin_memory=True, num_workers=4)

        self.test_dataset = PDLDataSet(test_data, mode='normal')
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.test_batch_size, shuffle=False)

    def train(self, ref_data: DataLoader, soft_decision_target: Tensor) -> None:
        ref_sample = dl_to_sampler(ref_data)
        self.model.to(self.device)
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            local_loss = torch.nn.CrossEntropyLoss()(output, target)

            if soft_decision_target is None:
                soft_decision_loss = 0
            else:
                with torch.no_grad():
                    extra_data_sampled, _, ref_idx = ref_sample()
                    extra_data_sampled = extra_data_sampled.to(self.device)
                soft_decision = F.softmax(self.model(extra_data_sampled), dim=1)

                soft_decision_target = soft_decision_target.to(self.device)
                soft_decision_loss = torch.mean(
                    -torch.sum(soft_decision_target[ref_idx, :] * torch.log(soft_decision + 1e-8), dim=1))

            loss = local_loss + self.lambda_ * soft_decision_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def infer_on_refdata(self, ref_loader: DataLoader) -> Tuple[Tensor, Tensor]:
        soft_decision_list = []
        ref_y = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, y, _) in enumerate(ref_loader):
                data = data.to(self.device)
                soft_decision = F.softmax(self.model(data), dim=1)
                soft_decision_list.append(soft_decision)
                ref_y.append(y)
        soft_decisions = torch.cat(soft_decision_list, dim=0)
        ref_y = torch.cat(ref_y, dim=0)
        return soft_decisions.detach().cpu(), ref_y.detach().cpu()

    def soft_assignment_to_hard_assignment(self, soft_decisions: Tensor) -> Tensor:
        with torch.no_grad():
            a = torch.argmax(soft_decisions, 1)
            hard_decisions = torch.zeros_like(soft_decisions)
            for i in range(hard_decisions.shape[0]):
                hard_decisions[i, a[i]] = 1
            return hard_decisions

    @torch.no_grad()
    def calculate_trust_weight_mat(self, soft_decision_list: List[Tensor]) -> Tensor:
        sd_eigen = soft_decision_list[self.id]
        sim_list = []
        for j in range(len(soft_decision_list)):
            sd_other = soft_decision_list[j]
            sim_ = self.sim(sd_other.to(self.device), sd_eigen.to(self.device))
            sim_list.append(sim_)

        self.trust_weights = torch.tensor(sim_list)
        self.trust_weights /= torch.sum(self.trust_weights)
        return self.trust_weights

    @torch.no_grad()
    def calculate_soft_decision_target(self, global_round: int, soft_decisions: List[Tensor]) -> Tuple[Tensor, Tensor]:
        match self.consensus:
            case ConsensusName.soft_assignment:
                match self.trust:
                    case TrustName.static:
                        if not self.trust_weights:
                            self.trust_weights = self.calculate_trust_weight_mat(soft_decisions)
                    case TrustName.dynamic:
                        if (global_round - self.pretraining_rounds) % self.trust_update_frequency == 0:
                            self.trust_weights = self.calculate_trust_weight_mat(soft_decisions)
                    case TrustName.naive:
                        self.trust_weights = torch.ones(self.num_clients, 1) / self.num_clients
                    case _:
                        raise UnknownNameCustomEnumException(self.trust, TrustName)

                weighted_soft_decisions = [self.trust_weights[i] * soft_decisions[i] for i in range(self.num_clients)]
                target = sum(weighted_soft_decisions)
                target = torch.nn.functional.normalize(target, p=1.0, dim=0)
                return target, self.trust_weights

            case ConsensusName.majority_vote:
                hard_labels = [self.soft_assignment_to_hard_assignment(soft_decisions[i]) for i in
                               range(self.num_clients)]
                target = sum(hard_labels)
                target = self.soft_assignment_to_hard_assignment(target)
                self.trust_weights = torch.ones(self.num_clients, 1) / self.num_clients
                return target, self.trust_weights
            case _:
                raise UnknownNameCustomEnumException(self.consensus, ConsensusName)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def test(self, metric: MetricName, mode: str = 'test') -> float:
        match mode:
            case 'train':
                loader = self.train_loader
            case 'test':
                loader = self.test_loader
            case _:
                raise NotImplementedError

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            match metric:
                case MetricName.acc:
                    preds = []
                    for batch_idx, (data, target) in enumerate(loader):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        pred_test = torch.sum(torch.argmax(self.model(data), dim=1) == target)
                        preds.append(pred_test.detach().cpu())
                    return np.sum(preds) / len(loader.dataset)
                case MetricName.bacc:
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
                case _:
                    raise UnknownNameCustomEnumException(metric, MetricName)

    def get_train_set_size(self) -> int:
        return self.train_loader.__len__()
