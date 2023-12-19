import argparse
from argparse import Namespace

import distributed


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # setup
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='none')
    parser.add_argument('--use_pretrained', default='gpt2', type=str)

    parser.add_argument('--num_clients', required=True, type=int)
    parser.add_argument('--num_classes', required=True, type=int)
    parser.add_argument('--num_global_rounds', default=2000, type=int)
    parser.add_argument('--acc_steps', default=4, type=int)
    parser.add_argument('--pretraining_rounds', default=10, type=int)
    parser.add_argument('--eval_freq', default=200, type=int)

    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lambda_', default=0.5, type=float)
    parser.add_argument('--sequence_length', default=512, type=int)
    parser.add_argument('--trust', type=str, help='none, dynamic, naive')
    parser.add_argument('--trust_freq', type=int, default=1)

    ##lora config
    parser.add_argument('--warmup_percent', default=0.04, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--scheduler', default='cos', choices=['linear', 'cos', 'none'])
    parser.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--grad_clip', default=0.0, type=float)  # default value is 1.0 in NanoGPT
    parser.add_argument('--distributed_backend', default=None, type=str, required=False,
                        choices=distributed.registered_backends())  # distributed backend type

    parser.add_argument('--lora_rank', default=4, type=int)
    parser.add_argument('--lora_alpha', default=32., type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--lora_mlp', action='store_true')
    parser.add_argument('--lora_embedding', action='store_true')
    parser.add_argument('--lora_causal_self_attention', action='store_true')
    parser.add_argument('--lora_freeze_all_non_lora', action='store_true')

    # dataset args
    parser.add_argument('--class_per_client', type=int, default=2)
    parser.add_argument('--niid', action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--partition', type=str, default='dir')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--ref', action='store_true')

    return parser.parse_args()
