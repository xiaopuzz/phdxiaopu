import numpy as np
import torch.nn.functional as F

from argparse import ArgumentParser
from copy import deepcopy
from data import get_data
from nets import *
from utils import *

class Summary:
    def __init__(self):
        self.loss = []
        self.y = []
        self.probs = []

    def update(self, loss, y, probs):
        self.loss.append(loss)
        self.y.append(y)
        self.probs.append(probs)

    def get_loss(self):
        return np.mean(self.loss)

    def get_auc(self):
        return roc_auc_score(np.concatenate(self.y), np.concatenate(self.probs))

def train_epoch(train_data, net, optimizer):
    summary = Summary()
    net.train()
    for x, y in train_data:
        x, y = x.cuda(), y.cuda()
        logits = net(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        probs = torch.sigmoid(logits)
        summary.update(loss.item(), y.data.cpu().numpy(), probs.data.cpu().numpy())
    return summary

def eval_epoch(eval_data, net):
    summary = Summary()
    net.eval()
    with torch.no_grad():
        for x, y in eval_data:
            x, y = x.cuda(), y.cuda()
            logits = net(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            probs = torch.sigmoid(logits)
            summary.update(loss.item(), y.data.cpu().numpy(), probs.data.cpu().numpy())
    return summary

def main(args):
    print(args)
    set_seed(args['seed'])
    train_data, val_data, test_data = get_data(args)
    net = MLP(train_data.num_features, args['h_dim']).cuda()
    optimizer = torch.optim.SGD(net.parameters(), args['lr_init'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['num_epochs'], eta_min=args['lr_final'])
    optimal_val_loss = np.inf
    optimal_weights = None
    for cur_epoch in range(args['num_epochs']):
        scheduler.step(cur_epoch)
        train_summary = train_epoch(train_data, net, optimizer)
        val_summary = eval_epoch(val_data, net)
        val_loss = val_summary.get_loss()
        if val_loss < optimal_val_loss:
            optimal_val_loss = val_loss
            optimal_weights = deepcopy(net.state_dict())
        if cur_epoch % 100 == 0:
            print(f'{get_time()}, epoch {cur_epoch}, train auc {train_summary.get_auc():.3f}, val_auc {val_summary.get_auc():.3f}')
    net.load_state_dict(optimal_weights)
    test_summary = eval_epoch(test_data, net)
    print(f'test_auc {test_summary.get_auc():.3f}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr_init', type=float, default=0.1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()
    args = vars(args)
    args['is_nn'] = True
    args['lr_final'] = args['lr_init'] / 1e3
    main(args)