import argparse
from typing import Tuple

from utils.types import DatasetName, TrustName, ConsensusName, SettingName, SimMeasureName, MetricName


# @todo finish correct parser

def get_args() -> Tuple[
    int, int, int, int, int, float, float, int, int, TrustName, ConsensusName, DatasetName,
    int, int, int, SimMeasureName, int, bool, SettingName, str, MetricName, int, bool
]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-expno", "--experiment_no", default=0, type=int)
    parser.add_argument("-seed", "--seed", default=11, type=int)
    parser.add_argument("-nc", "--num_clients", default=10, type=int)
    parser.add_argument("-gr", "--num_global_rounds", default=50, type=int)
    parser.add_argument('-le', "--num_local_epochs", default=5, type=int)
    parser.add_argument('-lr', "--learning_rate", default=5e-3, type=float)
    parser.add_argument('-lam', '--lambda_', default=0.5, type=float)
    parser.add_argument('-ncl', '--num_classes', default=10, type=int)
    parser.add_argument('-nch', '--num_channels', default=3, type=int)
    parser.add_argument('-trust', '--trust_update', default='dynamic', type=str, help='static, dynamic, naive')
    parser.add_argument('-consensus', '--consensus_mode', default='soft_assignment', type=str,
                        help='majority_voting, soft_assignment')
    parser.add_argument('-ds', '--dataset_name', type=str,
                        required=True, help=", ".join([j.__str__() for j in DatasetName]))
    parser.add_argument('-train_bs', '--train_batch_size', type=int, default=64)
    parser.add_argument('-ref_bs', '--ref_batch_size', type=int, default=256)
    parser.add_argument('-test_bs', '--test_batch_size', type=int, default=256)
    parser.add_argument('-sim', '--sim_measure', type=str, default='cosine', help='[true_label,cosine]')
    parser.add_argument('-prer', '--pretraining_rounds', type=int, default=5)
    parser.add_argument('-cmode', '--cmode', action='store_true')
    parser.add_argument('-setting', '--setting', type=str, default='normal', help='choose between [2sets, evil,normal]')
    parser.add_argument('-arch_name', '--arch_name', type=str, default='efficientnet-b0',
                        help='only when selects fed_isic')
    parser.add_argument('-metric', '--metric', type=str, default='acc', help='choose between bacc and acc')
    parser.add_argument('-trust_freq', '--trust_update_frequency', type=int, default=1,
                        help='how often should trust be updated')
    parser.add_argument('-graph', '--graph', action='store_true')
    args = parser.parse_args()

    return (args.experiment_no, args.seed, args.num_clients, args.num_global_rounds, args.num_local_epochs,
            args.learning_rate, args.lambda_, args.num_classes, args.num_channels, args.trust_update,
            args.consensus_mode, args.dataset_name, args.train_batch_size, args.ref_batch_size, args.test_batch_size,
            args.sim_measure, args.pretraining_rounds, args.cmode, args.setting, args.arch_name, args.metric,
            args.trust_update_frequency, args.graph)
