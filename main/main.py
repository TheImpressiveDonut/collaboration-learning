import numpy as np
import torch
from torch.utils.data import DataLoader

from models.models_utils import SharedData, transform_ref_data
from training.setting_selection import setting_selection
from parser import get_args
from training.trainer import Trainer
from utils.memory import load_dataset

(experiment_no, seed, num_clients, num_global_rounds, num_local_epochs, learning_rate, lambda_, num_classes,
 num_channels, trust_update, consensus_mode, dataset_name, train_batch_size, ref_batch_size,
 test_batch_size, sim_measure) = get_args()
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

train_data, test_data, ref_data = load_dataset('test')  # @todo do better path management

clients = setting_selection(train_data, test_data, args)
clients_sample_size = np.empty(args.num_clients, dtype=np.int64)
for key, val in clients.items():
    clients_sample_size[val] = val.get_train_set_size()

ref_X, ref_y = transform_ref_data(ref_data)
ref_loader = DataLoader(SharedData(ref_X, ref_y), batch_size=args.ref_batch_size,
                        shuffle=False, pin_memory=True, num_workers=0)

model_trainer = Trainer(clients, )
print('Training started:')
model_trainer.train(EPOCHS, )
print('Training finished:')
