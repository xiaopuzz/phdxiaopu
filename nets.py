import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_features, h_dim):
        super(MLP, self).__init__()
        self.fc_0 = nn.Linear(num_features, h_dim)
        self.fc_1 = nn.Linear(h_dim, h_dim)
        self.fc_2 = nn.Linear(h_dim, h_dim)
        self.output = nn.Linear(h_dim, 1)
        self.bn_0 = nn.BatchNorm1d(h_dim)
        self.bn_1 = nn.BatchNorm1d(h_dim)
        self.bn_2 = nn.BatchNorm1d(h_dim)

    def forward(self, x):
        x = x
        x = self.bn_0(torch.relu_(self.fc_0(x)))
        # x = F.dropout(x)
        x = self.bn_1(torch.relu_(self.fc_1(x)))
        # x = F.dropout(x)
        x = self.bn_2(torch.relu_(self.fc_2(x)))
        # x = F.dropout(x)
        return self.output(x).squeeze()