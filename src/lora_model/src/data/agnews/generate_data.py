import numpy as np
import os
import sys
import random
import torchtext
import sys
import json
import argparse
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset_utils import separate_data, split_data, save_file
from torchtext.data.utils import get_tokenizer
import tiktoken



parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default = 6,
                    help='number of clients')
parser.add_argument('--dir', type=str, help='directory for storing the data',
                    default = parent_dir + "/agnews/")
parser.add_argument('--niid',  action='store_false', help='sample non-iid data for each worker' )
parser.add_argument('--balance', action='store_true')
parser.add_argument('--partition',type=str,default='dir' )
parser.add_argument('--alpha',type=float ,default=0.5 ,help='needed when using dirichelet distr')


random.seed(1)
np.random.seed(1)
num_classes = 4


# Allocate data to users
def generate_agnews(dir_path, num_clients, num_classes, alpha ,niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train"
    test_path = dir_path + "test"

    # Get AG_News data
    trainset, testset = torchtext.datasets.AG_NEWS(root=dir_path+"rawdata")

    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)
    dataset_label = np.array(dataset_label)

    tokenizer = get_tokenizer('basic_english')
    tokenizer = tiktoken.get_encoding("gpt2")
    # raw_tokenized = tokenizer.encode_ordinary_batch(dataset_text)

    X, y, statistic = separate_data((dataset_text, dataset_label), num_clients, num_classes, alpha ,niid, balance, partition)
    train_data, test_data = split_data(X, y)
    tokenized_train_data = []
    tokenized_test_data = []
    for i in range(len(train_data)):
        train_combined_texts =  " ".join(train_data[i])
        test_combined_texts =  " ".join(test_data[i])
        tokenized_train_data.append(tokenizer.encode_ordinary(train_combined_texts))
        tokenized_test_data.append(tokenizer.encode_ordinary(test_combined_texts))

    save_file(config_path, train_path, test_path, tokenized_train_data, tokenized_test_data, num_clients, 
                num_classes, statistic, niid, balance, partition)


if __name__ == "__main__":
    args = parser.parse_args()
    niid = args.niid 
    balance = args.balance
    partition = args.partition
    alpha = args.alpha
    print("non iid:", args.niid)
    print("partition:", args.partition)
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    generate_agnews(args.dir, args.n_clients, num_classes, alpha, niid, balance, partition)