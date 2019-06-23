import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from utils import *

class StandardDataLoader(DataLoader):
    def __init__(self, x, y, batch_size, is_shuffle):
        super(StandardDataLoader, self).__init__(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y,
            dtype=torch.float)), batch_size=batch_size, shuffle=is_shuffle)
        self.num_features = x.shape[-1]

def normalize(x, mean, sd):
    x -= mean
    x /= sd
    return x

def parse(fpath):
    df = pd.read_csv(fpath)
    df.loc[df.population == 'EUR', 'population'] = 0
    df.loc[df.population == 'YRI', 'population'] = 1
    x = df.iloc[:, 1:-1].values
    y = df.population.values
    return x, y

def get_data(args):
    x_trainval, y_trainval = parse('data/uptrainingset.csv')
    x_test, y_test = parse('data/testingset.csv')
    # Shuffle
    idxs = np.random.permutation(len(x_trainval))
    x_trainval, y_trainval = x_trainval[idxs], y_trainval[idxs]
    num_train = int(len(x_trainval) * (1 - args['val_ratio']))
    # Split data
    x_train, y_train = x_trainval[:num_train], y_trainval[:num_train]
    x_val, y_val = x_trainval[num_train:], y_trainval[num_train:]
    # Normalize using train statistics
    x_mean, x_sd = x_train.mean(0), x_train.std(0)
    x_train = normalize(x_train, x_mean, x_sd)
    x_val = normalize(x_val, x_mean, x_sd)
    x_test = normalize(x_test, x_mean, x_sd)
    if args['is_nn']:
        train_data = StandardDataLoader(x_train, y_train, args['batch_size'], True)
        val_data = StandardDataLoader(x_val, y_val, args['batch_size'], False)
        test_data = StandardDataLoader(x_test, y_test, args['batch_size'], False)
        return train_data, val_data, test_data
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test

# def parse(fpath):
#     df = pd.read_csv(fpath)
#     df.drop('Sum', axis=1, inplace=True)
#     df.loc[df.population == 'EUR', 'population'] = 0
#     df.loc[df.population == 'YRI', 'population'] = 1
#     x = df.iloc[:, 2:].values
#     y = df.population.values
#     return x, y

# def save_combined():
#     print('Save combined')
#     fpaths = os.listdir('data')
#     x = []
#     y = []
#     max_num_cols = -np.inf
#     for fpath in fpaths:
#         x_elem, y_elem = parse(fpath)
#         num_cols = x_elem.shape[-1]
#         if num_cols > max_num_cols:
#             max_num_cols = num_cols
#         x.append(x_elem)
#         y.append(y_elem)
#     for i in range(len(x)):
#         x_elem = x[i]
#         num_rows, num_cols = x_elem.shape
#         pad_size = max_num_cols - num_cols
#         if pad_size > 0:
#             x[i] = np.hstack((x_elem, np.zeros((num_rows, max_num_cols - num_cols))))
#     x = np.concatenate(x)
#     y = np.concatenate(y)
#     save_file((x, y), 'data/data.pkl')