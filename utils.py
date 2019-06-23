import datetime
import numpy as np
import pickle
import torch

from sklearn.metrics import roc_auc_score

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def compute_acc(y, pred):
    pred_class = np.round(pred)
    return np.equal(y, pred_class, dtype='float').mean()

def get_time():
    return datetime.datetime.now().strftime('%H:%M:%S')

def save_file(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def str_to_bool(input):
    if input == 'True':
        return True
    elif input == 'False':
        return False
    else:
        return ValueError