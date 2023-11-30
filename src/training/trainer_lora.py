from typing import Dict

import numpy as np
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.client.client import Client
from src.client.client_lora import ClientLoRA
from src.utils.exceptions import UnknownNameCustomEnumException
from src.utils.types import MetricName


class TrainerLoRA(object):

    def __init__(self,
                 clients: Dict[int, ClientLoRA],
                 pretraining_rounds: int,
                 num_local_epochs: int,
                 metric: MetricName,
                 ) -> None:
        self.clients = clients
        self.pretraining_rounds = pretraining_rounds
        self.num_local_epochs = num_local_epochs
        self.metric_name = metric
        match metric:
            case MetricName.acc:
                self.metric = accuracy_score
            case MetricName.bacc:
                self.metric = balanced_accuracy_score
            case _:
                raise UnknownNameCustomEnumException(metric, MetricName)

        self.train_losses = []
        self.val_losses = []
        self.val_pps = []
        self.val_accs = []

    def train(self, global_epochs: int):
        global_epochs = tqdm(range(global_epochs), position=0, leave=False, desc='global epochs')
        prev_train_loss = 0
        prev_val_loss = 0

        for current_global_epoch in global_epochs:
            self.train_losses.append([])
            self.val_losses.append([])
            self.val_pps.append([])
            self.val_accs.append([])

            self.__global_epoch(current_global_epoch)

            train_loss = np.mean(self.train_losses[current_global_epoch])
            val_loss = np.mean(self.val_losses[current_global_epoch])
            if current_global_epoch == 0:
                prev_train_loss = train_loss
                prev_val_loss = val_loss
            global_epochs.set_description(
                f'global epochs (local accuracy: {train_loss:.5f}[{train_loss - prev_train_loss:+.5f}]'
                f' | global accuracy: {val_loss:.5f}[{val_loss - prev_val_loss:+.5f}])'
            )
            wandb.log({
                'global_test_accuracy': train_loss,
                'global_ref_accuracy': val_loss
            })

            wandb_dict = {'global_epoch': current_global_epoch}

            for idx, acc in enumerate(self.train_losses[current_global_epoch]):
                wandb_dict[f'{idx}_train_loss'] = acc

            for idx, acc in enumerate(self.val_losses[current_global_epoch]):
                wandb_dict[f'{idx}_val_loss'] = acc

            for idx, acc in enumerate(self.val_pps[current_global_epoch]):
                wandb_dict[f'{idx}_val_pp'] = acc

            for idx, acc in enumerate(self.val_accs[current_global_epoch]):
                wandb_dict[f'{idx}_val_acc'] = acc

            wandb.log(wandb_dict)

            prev_train_loss = train_loss
            prev_val_loss = val_loss

        for client in self.clients.values():
            client.distributed_backend.finalize()

        return self.train_losses, self.val_losses, self.val_pps, self.val_accs

    def __global_epoch(self, current_global_epoch: int):

        for idx, client in tqdm(self.clients.items(), position=1, leave=False, desc='clients'):
            for _ in tqdm(range(self.num_local_epochs), position=2, leave=False, desc='local epoch'):
                self.__local_epoch(client)

            val_acc, val_loss, val_perplexity = client.val()
            train_loss = client.loss.detach().cpu().item()
            self.train_losses[current_global_epoch].append(train_loss)
            self.val_losses[current_global_epoch].append(val_loss)
            self.val_pps[current_global_epoch].append(val_perplexity)
            self.val_accs[current_global_epoch].append(val_acc)

        self.__average_update()

    def __local_epoch(self, client: ClientLoRA):
        client.train()

    def __average_update(self):
        gradients = {}
        for client in self.clients.values():
            for name, param in client.model.named_parameters():
                if param.requires_grad:
                    if name in gradients:
                        gradients[name] += param.grad.clone()
                    else:
                        gradients[name] = param.grad.clone()

        for name in gradients.keys():
            gradients[name] /= len(self.clients.keys())

