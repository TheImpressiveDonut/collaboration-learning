from argparse import Namespace
from typing import Dict

import numpy as np
import wandb
from torch import Tensor
from tqdm import tqdm

from client.client import Client
from utils.exceptions import UnknownNameCustomEnumException
from utils.types import TrustName


class Trainer(object):

    def __init__(self, clients: Dict[int, Client], args: Namespace) -> None:
        self.clients = clients

        self.pretraining_rounds = args.pretraining_rounds
        self.acc_steps = args.acc_steps
        self.trust = args.trust
        self.trust_freq = args.trust_freq
        self.eval_freq = args.eval_freq

        self.train_losses = []
        self.val_losses = []
        self.val_pps = []
        self.val_accs = []

    def train(self, global_epochs: int):
        global_epochs = tqdm(range(1, global_epochs + 1), position=0, leave=False, desc='global epochs')
        prev_train_loss = 0
        prev_val_loss = 0

        for current_global_epoch in global_epochs:
            if current_global_epoch % self.eval_freq == 0:
                self.train_losses.append([])
                self.val_losses.append([])
                self.val_pps.append([])
                self.val_accs.append([])

            self.__global_epoch(current_global_epoch)

            if current_global_epoch % self.eval_freq == 0:

                train_loss = np.mean(self.train_losses[(current_global_epoch // self.eval_freq) - 1])
                val_loss = np.mean(self.val_losses[(current_global_epoch // self.eval_freq) - 1])

                global_epochs.set_description(
                    f'global epochs (local accuracy: {train_loss:.5f}[{train_loss - prev_train_loss:+.5f}]'
                    f' | global accuracy: {val_loss:.5f}[{val_loss - prev_val_loss:+.5f}])'
                )

                wandb_dict = {
                    'global_epoch': current_global_epoch,
                    'global_test_accuracy': train_loss,
                    'global_ref_accuracy': val_loss
                }

                for idx, acc in enumerate(self.train_losses[(current_global_epoch // self.eval_freq) - 1]):
                    wandb_dict[f'{idx}_train_loss'] = acc

                for idx, acc in enumerate(self.val_losses[(current_global_epoch // self.eval_freq) - 1]):
                    wandb_dict[f'{idx}_val_loss'] = acc

                for idx, acc in enumerate(self.val_pps[(current_global_epoch // self.eval_freq) - 1]):
                    wandb_dict[f'{idx}_val_pp'] = acc

                for idx, acc in enumerate(self.val_accs[(current_global_epoch // self.eval_freq) - 1]):
                    wandb_dict[f'{idx}_val_acc'] = acc

                wandb.log(wandb_dict)

                prev_train_loss = train_loss
                prev_val_loss = val_loss

        for client in self.clients.values():
            client.distributed_backend.finalize()

        return self.train_losses, self.val_losses, self.val_pps, self.val_accs

    def __global_epoch(self, current_global_epoch: int):

        for _, client in tqdm(self.clients.items(), position=1, leave=False, desc='clients'):
            self.__local_epoch(client, acc_steps=self.acc_steps)

            if current_global_epoch % self.eval_freq == 0:
                val_acc, val_loss, val_perplexity = client.val()
                train_loss = client.loss.detach().cpu().item()
                self.train_losses[(current_global_epoch // self.eval_freq) - 1].append(train_loss)
                self.val_losses[(current_global_epoch // self.eval_freq) - 1].append(val_loss)
                self.val_pps[(current_global_epoch // self.eval_freq) - 1].append(val_perplexity)
                self.val_accs[(current_global_epoch // self.eval_freq) - 1].append(val_acc)

        # trust update
        if current_global_epoch > self.pretraining_rounds and (current_global_epoch % self.trust_freq) == 0:
            match self.trust:
                case TrustName.static:
                    gradients = None
                case TrustName.naive:
                    gradients = self.__average_gradients()
                case _:
                    raise UnknownNameCustomEnumException(self.trust, TrustName)

            for _, clients in self.clients.items():
                clients.manual_grad_update(gradients)

    def __local_epoch(self, client: Client, acc_steps: int):
        client.train(acc_steps=acc_steps)

    def __average_gradients(self) -> Dict[str, Tensor]:
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

        return gradients
