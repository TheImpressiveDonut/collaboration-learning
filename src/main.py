from argparse import Namespace

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.client.client import Client
from src.client.client_utils import select_client_model_mode
from src.config.parser import get_dataset_main_args
from src.dataset_creation import create_dataset
from src.models.models_utils import SharedData, transform_ref_data
from src.training.trainer import Trainer
from src.utils.memory import load_dataset, save_results
from src.utils.types import Dataset


def main(args: Namespace, dataset: Dataset) -> None:
    print('datasets used:', args.dataset_name)
    print('setting:', args.setting)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set torch devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('******* the device used is:', device)
    if device.type == 'cuda':
        print('num of gpus:', torch.cuda.device_count())

    if args.wandb:
        wandb.init(project=args.wandb_project, name=f'{args.experiment_name}', config=vars(args))

    train_data, test_data, ref_data = dataset
    clients_model = select_client_model_mode(args)
    clients = {}
    for id, model_mode in clients_model.items():
        clients[id] = Client(id=id, model=model_mode[0], mode=model_mode[1], device=device,
                              train_data=train_data[id], test_data=test_data[id], ref_data=ref_data,
                              args=args)
    clients_sample_size = np.empty(args.num_clients, dtype=np.int64)
    for key, val in clients.items():
        clients_sample_size[key] = val.get_train_set_size()

    ref_X, ref_y = transform_ref_data(ref_data)
    ref_loader = DataLoader(SharedData(ref_X, ref_y), batch_size=args.ref_batch_size,
                            shuffle=False, pin_memory=True, num_workers=0)

    model_trainer = Trainer(clients, ref_loader, clients_sample_size, args.pretraining_rounds, args.num_local_epochs, args.metric)
    print('-' * 5, 'Training started', '-' * 5)
    trust_weights, test_accuracies, ref_accuracies = model_trainer.train(args.num_global_rounds)

    print('-' * 5, 'Saving results', '-' * 5)
    save_results(trust_weights, test_accuracies, ref_accuracies, args.dataset_name, args.experiment_name)

if  __name__ == '__main__':
    dataset_args, args = get_dataset_main_args()

    print('-' * 5, 'Loading dataset', '-' * 5)
    create_dataset.main(dataset_args)
    dataset = load_dataset(dataset_args)

    main(args, dataset)