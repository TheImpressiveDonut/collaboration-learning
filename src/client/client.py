import typing
from argparse import Namespace
from typing import Tuple, Dict

import numpy as np
import torch
from distributed.backend import DistributedBackend
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .model import GPTLoRA
from .utils import get_batch, eval


class Client(object):
    def __init__(self,
                 id: int, model: GPTLoRA, opt: Optimizer, distributed_backend: DistributedBackend,
                 scheduler: LRScheduler, train_data: np.ndarray, val_data: np.ndarray,
                 args: Namespace
                 ) -> None:
        self.id = id
        self.train_data = train_data
        self.val_data = val_data

        self.device = args.device
        self.optimizer = opt
        self.scheduler = scheduler
        self.distributed_backend = distributed_backend

        print(f"Compiling {id} client's model")
        self.model = typing.cast(GPTLoRA, torch.compile(model, dynamic=True))

        self.trust = args.trust
        self.lambda_ = args.lambda_
        self.grad_clip = args.grad_clip
        self.sequence_length = self.model.config.sequence_length
        self.batch_size = args.batch_size
        self.type_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    def train(self, acc_steps: int) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(self.train_data, self.sequence_length, self.batch_size, device=self.device)
            with self.type_ctx:
                with self.distributed_backend.get_context_for_microstep_forward(
                        model=self.model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    outputs = self.model(x, targets=y)

            loss = outputs['loss'] / acc_steps
            loss.backward()

        if self.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

    def val(self) -> Tuple[float, float, float]:
        self.model.eval()

        if self.distributed_backend.is_master_process():
            val_acc, val_loss, val_perplexity = eval(self.model, self.val_data, self.sequence_length, self.batch_size,
                                                     self.device, max_num_batches=24, ctx=self.type_ctx)
            return val_acc, val_loss, val_perplexity
        else:
            raise NotImplementedError

    def manual_grad_update(self, gradients: Dict[str, Tensor]):
        if not gradients is None:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.weight.grad = gradients[name]

            self.optimizer.step()
