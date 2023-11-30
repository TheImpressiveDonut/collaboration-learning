import argparse
from argparse import Namespace
from typing import Tuple

import torch

import src.lora_model.src.distributed as distributed
from src.dataset_creation import parser


def get_dataset_main_args() -> Tuple[Namespace, Namespace]:
    dataset_parser = parser.get_parser()

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('-expn', '--experiment_name', required=True, type=str)
    main_parser.add_argument('-seed', '--seed', default=11, type=int)
    main_parser.add_argument('--dataset_name', type=str, required=True, help='name of the dataset to generate')
    main_parser.add_argument('--wandb', action='store_true')
    main_parser.add_argument('--wandb_project', type=str, default='none')

    main_parser.add_argument('-train_bs', '--train_batch_size', type=int, default=64)
    main_parser.add_argument('-ref_bs', '--ref_batch_size', type=int, default=256)
    main_parser.add_argument('-test_bs', '--test_batch_size', type=int, default=256)

    main_parser.add_argument("-nc", "--num_clients", default=10, type=int)
    main_parser.add_argument('-ncl', '--num_classes', default=10, type=int)
    main_parser.add_argument("-gr", "--num_global_rounds", default=50, type=int)
    main_parser.add_argument('-le', "--num_local_epochs", default=5, type=int)

    main_parser.add_argument('-lr', "--learning_rate", default=5e-3, type=float)
    main_parser.add_argument('-lam', '--lambda_', default=0.5, type=float)

    main_parser.add_argument('-nch', '--num_channels', default=3, type=int)
    main_parser.add_argument('-trust', '--trust_update', default='dynamic', type=str, help='static, dynamic, naive')
    main_parser.add_argument('-consensus', '--consensus_mode', default='soft_assignment', type=str,
                             help='majority_voting, soft_assignment')

    main_parser.add_argument('-sim', '--sim_measure', type=str, default='cosine', help='[true_label,cosine]')
    main_parser.add_argument('-prer', '--pretraining_rounds', type=int, default=5)
    main_parser.add_argument('-cmode', '--cmode', action='store_true')
    main_parser.add_argument('-setting', '--setting', type=str, default='normal',
                             help='choose between [2sets, evil,normal]')
    main_parser.add_argument('-arch_name', '--arch_name', type=str, default='efficientnet-b0',
                             help='only when selects fed_isic')
    main_parser.add_argument('-metric', '--metric', type=str, default='acc', help='choose between bacc and acc')
    main_parser.add_argument('-trust_freq', '--trust_update_frequency', type=int, default=1,
                             help='how often should trust be updated')

    ##lora config
    main_parser.add_argument('--batch_size', default=50, type=int)
    main_parser.add_argument('--acc_steps', default=4, type=int)
    main_parser.add_argument('--iterations', default=15000, type=int)
    main_parser.add_argument('--lr', default=2e-3, type=float)
    main_parser.add_argument('--warmup_percent', default=0.02, type=float)
    main_parser.add_argument('--weight_decay', default=1e-3, type=float)
    main_parser.add_argument('--beta1', default=0.9, type=float)
    main_parser.add_argument('--beta2', default=0.95, type=float)
    main_parser.add_argument('--scheduler', default='cos', choices=['linear', 'cos', 'none'])
    main_parser.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'])
    main_parser.add_argument('--eval_freq', default=200, type=int)  # in iterations
    main_parser.add_argument('--results_base_folder', default="./exps", type=str)
    main_parser.add_argument('--grad_clip', default=0.0, type=float)  # default value is 1.0 in NanoGPT
    # Dataset params
    main_parser.add_argument('--vocab_size', default=50304, type=int)

    # Model params
    main_parser.add_argument('--use_pretrained', default="none",
                        type=str)  # 'none', 'gpt-2' or a path to the pretraind model
    main_parser.add_argument('--dtype', default=torch.bfloat16, type=torch.dtype)
    main_parser.add_argument('--eval_seq_prefix', default="The history of Switzerland ",
                        type=str)  # prefix used to generate sequences
    # Distributed args
    main_parser.add_argument('--distributed_backend', default=None, type=str, required=False,
                        choices=distributed.registered_backends())  # distributed backend type

    main_parser.add_argument('--lora_rank', default=4, type=int)
    main_parser.add_argument('--lora_alpha', default=32., type=float)
    main_parser.add_argument('--lora_dropout', default=0.1, type=float)
    main_parser.add_argument('--lora_mlp', action='store_true')
    main_parser.add_argument('--lora_embedding', action='store_true')
    main_parser.add_argument('--lora_causal_self_attention', action='store_true')
    main_parser.add_argument('--lora_freeze_all_non_lora', action='store_true')

    main_parser.add_argument('args', nargs=argparse.REMAINDER)

    dataset_args, unknown = dataset_parser.parse_known_args()
    m_args = ['--dataset_name', dataset_args.dataset_name]
    m_args.extend(unknown)
    return dataset_args, main_parser.parse_args(m_args)
