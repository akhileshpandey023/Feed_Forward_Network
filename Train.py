#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import logging
import os
from utils import check_create_path, check_file
import pandas as pd
from FeedForwardNN import FeedforwardNN
import pickle 


## get arguments 
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type = float)
parser.add_argument('--momentum', type = float)
parser.add_argument('--num_hidden', type = int)
parser.add_argument('--sizes')
parser.add_argument('--activation', type = str)
parser.add_argument('--loss', type = str)
parser.add_argument('--opt', type = str)
parser.add_argument('--batch_size', type = int)
parser.add_argument('--epochs', type = int)
parser.add_argument('--anneal')
parser.add_argument('--save_dir', type = str)
parser.add_argument('--expt_dir', type = str)
parser.add_argument('--train', type = str)
parser.add_argument('--val', type = str)
parser.add_argument('--test', type = str)

args = parser.parse_args()


lr = args.lr
momentum = args.momentum
sizes = str.split(args.sizes, ',')
sizes = [int(i) for i in sizes]
activation = args.activation
loss = args.loss
optimization = args.opt
batch_size = args.batch_size
epochs = args.epochs
anneal = True if args.anneal == 'True' else False

save_dir_path = check_create_path(args.save_dir)
expt_dir_path = check_create_path(args.expt_dir)
train_path = args.train
test_path = args.test
valid_path = args.val

check_file(train_path)
check_file(test_path)
check_file(valid_path)

list_sizes = str.split(args.sizes, ',')
assert len(list_sizes) == args.num_hidden, 'Number of hidden layers and number of sizes should be same'

## Change directory to the model saving location
os.chdir(save_dir_path)

train_data = pd.read_csv(train_path, header=0, index_col= 0)
validation_data = pd.read_csv(valid_path, header=0, index_col= 0)

n_inputs = 784
n_output = 10

nn = FeedforwardNN(n_inputs, sizes, n_output, loss, activation)

X_train = np.array(train_data[['feat' + str(i) for i in range(784)]]).reshape(train_data.shape[0], 784)
X_train_norm = nn.normalise_fit(X_train)

y_train = np.array(train_data['label']).reshape(-1,1)

X_validation = np.array(validation_data[['feat' + str(i) for i in range(784)]]).reshape(validation_data.shape[0], 784)
X_validation_norm = nn.normalise_transform(X_validation)

y_validation = np.array(validation_data['label']).reshape(-1,1)

gamma = momentum
#gamma = 0.9
beta_1 = 0.9
beta_2 = 0.999


nn_model = nn.fit(X_train_norm, y_train, X_validation_norm, y_validation, batch_size, epochs, lr, optimization, gamma, beta_1, beta_2, anneal)

filename = save_dir_path + "nn_model.pkl"
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, 'wb') as f:
    pickle.dump(nn_model, f)
