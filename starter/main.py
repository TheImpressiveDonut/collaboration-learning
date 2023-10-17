import numpy as np
import torch
from torch.utils.data import DataLoader

from client.client import Client
from models.models_utils import SharedData, transform_ref_data
from starter.parser import get_args
from training.setting_selection import setting_selection
from training.trainer import Trainer
from utils.memory import load_dataset, save_results

(experiment_no, seed, num_clients, num_global_rounds, num_local_epochs, learning_rate, lambda_, num_classes,
 num_channels, trust_update, consensus_mode, dataset_name, train_batch_size, ref_batch_size,
 test_batch_size, sim_measure, pretraining_rounds, cmode, setting, arch_name, metric,
 trust_update_frequency, find_collaborators) = get_args()
print('datasets used:', dataset_name)
print('setting:', setting)

# set random seed
torch.manual_seed(seed)
np.random.seed(seed)

# set torch devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('******* the device used is:', device)
if device.type == 'cuda':
    print('num of gpus:', torch.cuda.device_count())

print('-' * 5, 'Loading dataset', '-' * 5)
train_data, test_data, ref_data = load_dataset(dataset_name)  # @todo sanity check config

clients_model = setting_selection(setting, dataset_name, num_clients)
clients = {}
for key, val in clients_model.items():
    clients[key] = Client(worker_index=key, dataset_name=dataset_name, model_name=val[0],
                          train_data=train_data[key], test_data=test_data[key], ref_data=ref_data,
                          num_classes=num_classes, num_channels=num_channels, num_clients=num_clients,
                          eff_net_arch_name=arch_name, learning_rate=learning_rate, lambda_=lambda_,
                          device=device, train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                          sim_measure=sim_measure, cosine_regularized=cmode, mode=val[1], consensus_mode=consensus_mode,
                          trust_update=trust_update, trust_update_frequency=trust_update_frequency,
                          pretraining_rounds=pretraining_rounds)
clients_sample_size = np.empty(num_clients, dtype=np.int64)
for key, val in clients.items():
    clients_sample_size[key] = val.get_train_set_size()

ref_X, ref_y = transform_ref_data(ref_data)
ref_loader = DataLoader(SharedData(ref_X, ref_y), batch_size=ref_batch_size,
                        shuffle=False, pin_memory=True, num_workers=0)

model_trainer = Trainer(clients, ref_loader, clients_sample_size, pretraining_rounds, num_local_epochs, metric)
print('-' * 5, 'Training started', '-' * 5)
trust_weights, test_accuracies, ref_accuracies = model_trainer.train(num_global_rounds)

print('-' * 5, 'Saving results', '-' * 5)
save_results(trust_weights, test_accuracies, ref_accuracies, dataset_name, experiment_no)

