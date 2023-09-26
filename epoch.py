import argparse

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from client.client import Client
from models.data_utils import SharedData, save_results
from utils.memory import load_dataset
from utils.types import DatasetName, Clients


# model training

def train_one_round(clients: Clients, round: int, args: argparse.Namespace):
    train_accs, test_accs, ref_accs = [], [], []
    soft_decisions, trust_weight, trust_weight_tmp = [], [], []

    for i in range(args.num_clients):
        if round < args.pretraining_rounds:
            soft_decision_target = None
        else:
            soft_decision_target, trust_weights = clients[i].calculate_soft_decision_target(
                soft_decisions[round - 1], train_accs[round - 1], clients_sample_size, global_round=round)
            trust_weight_tmp.append(trust_weights)

        for epoch in range(args.num_local_epochs):
            clients[i].train(epoch, ref_loader, soft_decision_target)

        train_accs.append(clients[i].test(mode='train'))
        soft_decision_tmp = clients[i].infer_on_refdata(ref_loader)
        soft_decision.append(soft_decision_tmp)

        ref_pred = torch.argmax(soft_decision_tmp, dim=1)
        if args.metric == 'acc': # @todo enum for metric accuracy
            ref_accs.append(sum(ref_pred == ref_y) / len(ref_y))
        elif args.metric == 'bacc':
            ref_accs.append(balanced_accuracy_score(ref_y, ref_pred))

        test_accs.append(clients[i].test(args.metric))
        tmp_path = 'models/model_weights/worker_' + str(i) + '.pt'

    if len(trust_weight_tmp) > 0:
        trust_weight_dict[round] = torch.stack(trust_weight_tmp)

    print('round %i finished' % round)
    print('local accuracy after round', str(round), ':', np.mean(test_accuracy_dict[round]))
    print('local accuracy:', test_accuracy_dict[round])
    print('global accuracy after round', str(round), ':', np.mean(ref_accuracy_dict[round]))
    print('global acc:', ref_accuracy_dict[round])

    if save_results(trust_weight_dict, test_accuracy_dict, ref_accuracy_dict, res_path, args) == 1:
        print('saved file successfully! The results are under', args.res_path)

    return train_accs, test_accs, ref_accs, soft_decisions