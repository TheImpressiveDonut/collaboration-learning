import argparse

from utils.types import DatasetName


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-expno", "--experiment_no", default=0, type=int)
    parser.add_argument("-seed", "--seed", default=11, type=int)
    parser.add_argument("-nc", "--num_clients", default=10, type=int)
    parser.add_argument("-gr", "--num_global_rounds", default=50, type=int)
    parser.add_argument('-le', "--num_local_epochs", default=5, type=int)
    parser.add_argument('-lr', "--learning_rate", default=5e-3, type=float)
    parser.add_argument('-mom', "--momentum", default=0.9)
    parser.add_argument('-lam', '--lambda_', default=0.5, type=float)
    parser.add_argument('-ncl', '--num_classes', default=10, type=int)
    parser.add_argument('-nch', '--num_channels', default=3, type=int)
    parser.add_argument('-trust', '--trust_update', default='dynamic', type=str, help='static, dynamic, naive')
    parser.add_argument('-consensus', '--consensus_mode', default='soft_assignment', type=str,
                        help='majority_voting, soft_assignment')
    parser.add_argument('-ds', '--dataset_name', type=str,
                        required=True, help=", ".join([j.__str__() for j in DatasetName]))
    parser.add_argument('-gpuids', '--device_ids', type=list,
                        default=[0])  # @todo useless param ?
    parser.add_argument('-device', '--device', type=str,
                        default='cuda')  # @todo useless param ?
    parser.add_argument('-train_bs', '--train_batch_size', type=int, default=64)
    parser.add_argument('-ref_bs', '--ref_batch_size', type=int, default=256)
    parser.add_argument('-test_bs', '--test_batch_size', type=int, default=256)
    parser.add_argument('-sim', '--sim_measure', type=str, default='cosine', help='[true_label,cosine]')
    parser.add_argument('-prer', '--pretraining_rounds', type=int, default=5)
    parser.add_argument('-cmode', '--cmode', type=str, default='regularized')
    parser.add_argument('-setting', '--setting', type=str, default='normal', help='choose between [2sets, evil,normal]')
    # parser.add_argument('-sampler','--sample_ratio',type=float,default =1.0,help='sample shared data to fasten training process')
    parser.add_argument('-arch_name', '--arch_name', type=str, default='efficientnet-b0',
                        help='only when selects fed_isic')
    parser.add_argument('-metric', '--metric', type=str, default='acc', help='choose between bacc and acc')
    parser.add_argument('-trust_freq', '--trust_update_frequency', type=int, default=1,
                        help='how often should trust be updated')
    return parser
