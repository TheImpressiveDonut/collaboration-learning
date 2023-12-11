import inspect
import random
from argparse import Namespace
from typing import Tuple, List

import numpy as np
import torch
import wandb

import distributed
from client.client import Client
from client.model import GPTLoRA
from data.utils import get_dataset
from parser import get_args
from utils.types import ClientsDataStatistics
from trainer import Trainer


def main(args: Namespace, dataset: Tuple[List[np.ndarray], List[np.ndarray]], stats: ClientsDataStatistics) -> None:
    print('datasets used:', args.dataset)
    print('trust:', args.trust)

    torch.backends.cuda.matmul.allow_tf32 = True  # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set torch devices
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('******* the device used is:', args.device)
    if args.device.type == 'cuda':
        print('num of gpus:', torch.cuda.device_count())

    if args.wandb:
        wandb.init(project=args.wandb_project, group=f'{args.experiment_name}', config=vars(args))
        unique = []
        for i in range(args.num_clients):
            for val, _ in stats[i]:
                if val not in unique:
                    unique.append(val)
        unique = sorted(unique)
        data = []
        for i in range(args.num_clients):
            data.append([0 for _ in range(len(unique) - 1)])
            for val, count in stats[i]:
                data[i][val] = count
        stats = wandb.Table(columns=unique, data=data)
        wandb.log({'Statistics': stats})


    train_data, val_data = dataset

    clients = {}
    for id in range(args.num_clients):
        distributed_backend = distributed.make_backend_from_args(args)
        args_db = distributed_backend.get_adjusted_args_for_process(args)
        model = GPTLoRA.from_pretrained(args.use_pretrained, args_db, verbose=(id == 0)).to(args_db.device)
        model.crop_sequence_length(args_db.sequence_length)
        model = distributed_backend.transform_model(model)

        group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
        param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
        optimized_params_cnt = 0
        optimized_required_params_cnt = 0
        for g in group_specs:
            params = []
            for p_name in g["params"]:
                translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
                params += [param_name_mapping[p_name] for p_name in translated_p_names]
            g["params"] = params
            optimized_params_cnt += sum([p.numel() for p in g["params"]])
            optimized_required_params_cnt += sum([p.numel() if p.requires_grad else 0 for p in g["params"]])
        if id == 0:
            print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
            print("number of required optimized parameters: %.2fM" % (optimized_required_params_cnt / 1e6,))

        for g in group_specs:
            params = [p for p in g["params"] if p.requires_grad]
            g["params"] = params

        if args_db.opt == 'adamw':
            use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters)
            if id == 0:
                print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            opt = torch.optim.AdamW(group_specs, lr=args_db.lr, betas=(args_db.beta1, args_db.beta2),
                                    weight_decay=args.weight_decay, **extra_args)
        else:
            opt = torch.optim.SGD(group_specs, lr=args_db.lr, momentum=0.9, weight_decay=args_db.weight_decay)

        if args_db.scheduler != 'none':
            if args_db.scheduler in ['cos', 'linear']:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=opt, max_lr=args_db.lr,
                    total_steps=args.num_global_rounds,
                    pct_start=args_db.warmup_percent,
                    anneal_strategy=args_db.scheduler,
                    cycle_momentum=False, div_factor=1e2,
                    final_div_factor=.05
                )
            else:
                raise NotImplementedError(f"Unknown scheduler type: {args_db.scheduler}.")
        else:
            scheduler = None

        clients[id] = Client(
            id=id, model=model, opt=opt, distributed_backend=distributed_backend, scheduler=scheduler,
            train_data=train_data[id], val_data=val_data[id],
            args=args_db
        )

    trainer = Trainer(clients, args)
    print('-' * 5, 'Training started', '-' * 5)
    train_losses, val_losses, val_pps, val_accs = trainer.train(args.num_global_rounds)

    print('-' * 5, 'Training finished', '-' * 5)


if __name__ == '__main__':
    args = get_args()

    train_data, val_data, stats = get_dataset(args)

    print(f'Total number of train samples: {sum([len(a) for a in train_data])}')
    print(f'Total number of validation samples: {sum([len(a) for a in val_data])}')
    print(f'Num training tokens: {[len(a) for a in train_data]}')
    print(f'Num validation tokens: {[len(a) for a in val_data]}')
    for client in range(args.num_clients):
        print(f'Client {client}\t Size of data: {train_data[client].shape[0]}')
        print(f'\t\t Samples of labels: {[i for i in stats[client]]}')

    main(args, (train_data, val_data), stats)
