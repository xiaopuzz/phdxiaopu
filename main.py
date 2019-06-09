import numpy as np

from argparse import ArgumentParser
from nets import *
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from utils import *

def compute_acc(y, pred, pred_thold=0.5):
    pred_class = (pred > pred_thold).astype('int')
    return np.equal(y, pred_class, dtype='float').mean()

def normalize(x_train, x_test):
    x_mean = x_train.mean(0)
    x_sd = x_train.std(0)
    x_train -= x_mean
    x_train /- x_sd
    x_test -= x_mean
    x_test /= x_sd
    return x_train, x_test

def main(args):
    x_trainval = np.loadtxt('uptrainingset.csv', delimiter=',', dtype='float32')
    x_test = np.loadtxt('testingset.csv', delimiter=',', dtype='float32')
    y_trainval = x_trainval[:, -1]
    x_trainval = x_trainval[:, :-1]
    y_test = x_test[:, -1]
    x_test = x_test[:, :-1]
    y_trainval = y_trainval[:, None]
    y_test = y_test[:, None]
    idxs = np.random.permutation(len(x_trainval))
    x_trainval, y_trainval = x_trainval[idxs], y_trainval[idxs]
    x_trainval, x_test = normalize(x_trainval, x_test)
    # num_train = int(len(x_trainval) * 0.8)
    # x_train, y_train = x_trainval[:num_train], y_trainval[:num_train]
    # x_val, y_val = x_trainval[num_train:], y_trainval[num_train:]
    train_data = DataLoader(TensorDataset(torch.tensor(x_trainval), torch.tensor(y_trainval)),
        batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
        batch_size=args.batch_size)
    net = MLP().cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.lr_init, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr_final)
    for cur_epoch in range(args.num_epochs):
        scheduler.step(cur_epoch)
        net.train()
        loss_list = []
        for x, y in train_data:
            x, y = x.cuda(), y.cuda()
            logits = net(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        net.eval()
        y_list = []
        probs_list = []
        with torch.no_grad():
            for x, y in test_data:
                x, y = x.cuda(), y.cuda()
                probs = torch.sigmoid(net(x))
                y_list.append(y.data.cpu().numpy())
                probs_list.append(probs.data.cpu().numpy())
        y_list = np.concatenate(y_list)
        probs_list = np.concatenate(probs_list)
        test_auc = roc_auc_score(y_list, probs_list)
        test_acc = compute_acc(y_list, probs_list)
        if cur_epoch % 100 == 0:
            print(f'epoch {cur_epoch}, auc {test_auc:.3f}, acc {test_acc:.3f}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr_init', type=float, default=0.01)
    parser.add_argument('--lr_final', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    main(args)