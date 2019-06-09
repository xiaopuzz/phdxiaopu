import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_0 = nn.Linear(65, 256)
        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 1)

    def forward(self, x):
        out = x
        out = torch.relu_(self.fc_0(out))
        out = F.dropout(out)
        out = torch.relu_(self.fc_1(out))
        out = F.dropout(out)
        return self.fc_2(out)