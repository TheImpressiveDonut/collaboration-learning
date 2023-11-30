import inspect
import random
from argparse import Namespace

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

import lora_model.src.distributed as distributed
from src.client.client_lora import ClientLoRA
from src.config.lora_parser import get_args
from src.config.parser import get_dataset_main_args
from src.dataset_creation import create_dataset
from src.lora_model.src.data.utils import get_dataset
from src.lora_model.src.models.lora import GPTLoRA
from src.models.models_utils import SharedData, transform_ref_data
from src.training.trainer import Trainer
from src.training.trainer_lora import TrainerLoRA
from src.utils.memory import load_dataset, save_results
from src.utils.types import Dataset


def main(args: Namespace, dataset) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True  # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    print('datasets used:', args.dataset_name)
    print('setting:', args.setting)

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set torch devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('******* the device used is:', device)
    if device.type == 'cuda':
        print('num of gpus:', torch.cuda.device_count())

    if args.wandb:
        wandb.init(project=args.wandb_project, name=f'{args.experiment_name}', config=vars(args))

    train_data, val_data = dataset

    clients = {}
    for id in range(args.num_clients):
        distributed_backend = distributed.make_backend_from_args(args)
        args_db = distributed_backend.get_adjusted_args_for_process(args)

        model = GPTLoRA.from_pretrained(args.use_pretrained, args_db).to(device)
        model = distributed_backend.transform_model(model)
        group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
        param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
        optimized_params_cnt = 0
        for g in group_specs:
            params = []
            for p_name in g["params"]:
                translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
                params += [param_name_mapping[p_name] for p_name in translated_p_names]
            g["params"] = params
            optimized_params_cnt += sum([p.numel() for p in g["params"]])
        print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
        if args_db.opt == 'adamw':
            use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters)
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            opt = torch.optim.AdamW(group_specs, lr=args_db.lr, betas=(args_db.beta1, args_db.beta2),
                                    weight_decay=args.weight_decay, **extra_args)
        else:
            opt = torch.optim.SGD(group_specs, lr=args_db.lr, momentum=0.9, weight_decay=args_db.weight_decay)

        if args_db.scheduler != 'none':
            if args_db.scheduler in ['cos', 'linear']:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args_db.lr,
                                                                total_steps=args_db.iterations,
                                                                pct_start=args_db.warmup_percent,
                                                                anneal_strategy=args_db.scheduler,
                                                                cycle_momentum=False, div_factor=1e2,
                                                                final_div_factor=.05)
            else:
                raise NotImplementedError(f"Unknown scheduler type: {args_db.scheduler}.")
        else:
            scheduler = None

        clients[id] = ClientLoRA(id=id, model=model, opt=opt, distributed_backend=distributed_backend,
                                 scheduler=scheduler, acc_steps=args_db.acc_steps,
                                 device=device,
                                 train_data=train_data, val_data=val_data,
                                 args=args_db)

    model_trainer = TrainerLoRA(clients, args.pretraining_rounds, args.num_local_epochs,
                                args.metric)
    print('-' * 5, 'Training started', '-' * 5)
    trust_weights, test_accuracies, ref_accuracies = model_trainer.train(args.num_global_rounds)

    print('-' * 5, 'Saving results', '-' * 5)
    save_results(trust_weights, test_accuracies, ref_accuracies, args.dataset_name, args.experiment_name)


if __name__ == '__main__':
    args = get_args()

    data = get_dataset(args)

    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")

    main(args, (data['train'], data['val']))
