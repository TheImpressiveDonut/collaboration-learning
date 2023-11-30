import argparse
from argparse import ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--num_clients', type=int, required=True,
                        help='number of clients')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='number of classes')
    parser.add_argument('--class_per_client', type=int, default=2,
                        help='class per client')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='name of the dataset to generate')
    parser.add_argument('--niid', action='store_true',
                        help='sample non-idd data for each worker')
    parser.add_argument('--balance', action='store_true',
                        help='data balanced between clients')
    parser.add_argument('--partition', type=str, default='dir',
                        help='how to split data for different clients')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='needed when using Dirichlet distribution')
    parser.add_argument('--ref', action='store_true',
                        help='generate reference data')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch_size')
    parser.add_argument('--train_size', type=float, default=0.75,
                        help='train_size')

    return parser
