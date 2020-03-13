import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import EdgeConv, MessagePassing

import torch


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))
        self.con_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))

        # Final MLP for regression.
        self.fc1 = Lin(1 * dim, dim)
        self.fc2 = Lin(dim, dim)
        self.fc3 = Lin(dim, dim)
        self.fc4 = Lin(dim, 1)

    def forward(self, data):
        n = self.var_mlp(data.var_node_features)
        e = self.con_mlp(data.con_node_features)

        x = e.new_zeros((data.node_types.size(0), n.size(-1)))
        x = x.scatter_(0, data.assoc_var.view(-1, 1).expand_as(n), n)
        x = x.scatter_(0, data.assoc_con.view(-1, 1).expand_as(e), e)

        xs = [x]


        x = xs[-1]
        x = x[data.assoc_var]

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.fc4(x))

        return x.squeeze(-1)
