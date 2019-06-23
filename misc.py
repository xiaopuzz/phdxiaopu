import xgboost as xgb

from argparse import ArgumentParser
from data import get_data
from sklearn.linear_model import LogisticRegression
from utils import *

class OptimalHyperparams:
    def __init__(self, is_maximizing):
        self.is_maximizing = is_maximizing
        self.hyperparams = []
        self.scores = []

    def update(self, hyperparams, score):
        self.hyperparams.append(hyperparams)
        self.scores.append(score)

    def get_optimal(self):
        if self.is_maximizing:
            idx = np.argmax(self.scores)
        else:
            idx = np.argmin(self.scores)
        return self.hyperparams[idx], self.scores[idx]

def train_xgb(data, eta, max_depth):
    param = {'max_depth': max_depth, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(data, 'train')]
    num_round = 100
    model = xgb.train(param, data, num_round, evallist, verbose_eval=False)
    return model

def main(args):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(args)
    # Stores optimal hyperparams
    optimal_hyperparams = OptimalHyperparams(args['is_maximizing'])
    if args['model_type'] == 'log_reg':
        # Hyperparams for logistic regression
        l2_regs = [1e-4, 1e-3, 0.01, 0.1]
        # Iterate over all hyperparam configurations
        for l2_reg in l2_regs:
            # Fit training set
            model = LogisticRegression(C=1 / l2_reg, solver='lbfgs').fit(x_train, y_train)
            pred_val = model.predict_proba(x_val)[:, 1]
            val_auc = roc_auc_score(y_val, pred_val)
            # Store hyperparams, val_auc pair
            optimal_hyperparams.update(l2_reg, val_auc)
        # Get the hyperparams which performed best on val data
        optimal_l2_reg, optimal_val_auc = optimal_hyperparams.get_optimal()
        # Combine train and val data to maximize data, and fit with optimal hyperparams
        x_trainval, y_trainval = np.concatenate((x_train, x_val), axis=0), np.concatenate((y_train, y_val))
        model = LogisticRegression(C=1 / optimal_l2_reg, solver='lbfgs').fit(x_trainval, y_trainval)
        # Evaluate on test set
        pred_test = model.predict_proba(x_test)[:, 1]
        test_auc = roc_auc_score(y_test, pred_test)
        print(f'logistic regression: val auc {optimal_val_auc:.3f}, test auc {test_auc:.3f}')
    elif args['model_type'] == 'xgb':
        # Hyperparams for xgb: https://xgboost.readthedocs.io/en/latest/parameter.html
        etas = [0, 0.25, 0.5, 0.75, 1]
        max_depths = [2, 4, 8, 16]
        # xgb requires data to be in this format
        train_data = xgb.DMatrix(x_train, label=y_train)
        val_data = xgb.DMatrix(x_val, label=y_val)
        test_data = xgb.DMatrix(x_test, label=y_test)
        # Iterate over all hyperparam configurations
        for eta, max_depth in zip(etas, max_depths):
            # Fit training set
            model = train_xgb(train_data, eta, max_depth)
            pred_val = model.predict(val_data)
            val_auc = roc_auc_score(y_val, pred_val)
            # Store hyperparams, val_auc pair
            optimal_hyperparams.update((eta, max_depth), val_auc)
        # Get the hyperparams which performed best on val data
        (optimal_eta, optimal_max_depth), optimal_val_auc = optimal_hyperparams.get_optimal()
        # Combine train and val data to maximize data, and fit with optimal hyperparams
        x_trainval, y_trainval = np.concatenate((x_train, x_val), axis=0), np.concatenate((y_train, y_val))
        trainval_data = xgb.DMatrix(x_trainval, label=y_trainval)
        model = train_xgb(trainval_data, optimal_eta, optimal_max_depth)
        # Evaluate on test set
        pred_test = model.predict(test_data)
        test_auc = roc_auc_score(y_test, pred_test)
        print(f'xgb: val auc {optimal_val_auc:.3f}, test auc {test_auc:.3f}')
    else:
        raise ValueError

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--is_maximizing', type=str_to_bool, default='True')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()
    args = vars(args)
    args['is_nn'] = False
    main(args)