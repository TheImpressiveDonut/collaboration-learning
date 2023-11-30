from argparse import Namespace
from typing import Tuple

import torch

from src.client.client_utils import CosineSimilarity
from src.lora_model.src.optim.utils import get_batch, eval
from src.utils.exceptions import UnknownNameCustomEnumException
from src.utils.types import MetricName
from src.utils.types import SimMeasureName


class ClientLoRA(object):
    def __init__(self,
                 id: int, model, opt, distributed_backend, scheduler, acc_steps,
                 device: torch.device,
                 train_data, val_data,
                 args: Namespace
                 ) -> None:
        self.id = id
        self.device = device
        print(f'Compiling model {id}')
        self.model = torch.compile(model, dynamic=True)
        self.optimizer = opt
        self.scheduler = scheduler
        self.distributed_backend = distributed_backend

        self.consensus = args.consensus_mode
        self.trust = args.trust_update

        self.lambda_ = args.lambda_
        self.num_clients = args.num_clients
        self.trust_update_frequency = args.trust_update_frequency
        self.pretraining_rounds = args.pretraining_rounds
        self.trust_weights = None  # Tensor of size (num_clients, 1)

        match args.sim_measure:
            case SimMeasureName.cosine:
                self.sim = CosineSimilarity(args.cmode)
            case SimMeasureName.true_label:
                raise NotImplementedError
            case _:
                raise UnknownNameCustomEnumException(args.sim_measure, SimMeasureName)

        self.train_data = train_data
        self.val_data = val_data

        # lora specific
        self.loss = None
        self.acc_steps = acc_steps
        self.grad_clip = args.grad_clip
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.type_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)  # extra_args.dtype)

    def global_grad_update(self, grad):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.weight.grad = grad[name]

        self.optimizer.step()


    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for microstep_idx in range(self.acc_steps):  # gradient accumulation
            x, y = get_batch(self.train_data, self.sequence_length, self.batch_size, device=self.device)
            with self.type_ctx:
                with self.distributed_backend.get_context_for_microstep_forward(
                        model=self.model, microstep_idx=microstep_idx, gradient_accumulation_steps=self.acc_steps):
                    outputs = self.model(x, targets=y)

            loss = outputs['loss'] / self.acc_steps
            self.loss = loss * self.acc_steps
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
