import numpy as np
import torch

from main.model_selection import select_clients_model
from main.parser import get_parser

args = get_parser().parse_args()
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

clients = select_clients_model(args)
clients_sample_size = np.empty(args.num_clients)

train_data, test_data, ref_data = load_dataset(config)

ref_X, ref_y =
ref_loader = DataLoader(SharedData(ref_X, ref_y), batch_size=args.ref_batch_size,
                        shuffle=False, pin_memory=True, num_workers=0)


soft_decision_dict = []
test_accuracy_dict = []
ref_accuracy_dict = []
train_acc_dict = []
trust_weight_dict = []

for round in range(args.num_global_rounds):
