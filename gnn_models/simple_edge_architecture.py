import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import EdgeConv, MessagePassing

import torch


class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out):
        super(EdgeConv, self).__init__(aggr='mean')
        self.mlp = Seq(Lin(2 * F_in + 2, F_out), ReLU(), Lin(F_out, F_out))

    def forward(self, x, edge_index, edge_types):
        return self.propagate(edge_index, x=x, edge_types=edge_types)  # shape [N, F_out]

    def message(self, x_i, x_j, edge_types):
        # edge_features = torch.cat([x_i, x_j - x_i, edge_types], dim=1)  # shape [E, 2 * F_in]
        edge_features = torch.cat([x_i, x_j, edge_types], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))
        self.con_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))

        self.conv1 = EdgeConv(dim, dim)
        self.conv2 = EdgeConv(dim, dim)
        self.conv3 = EdgeConv(dim, dim)

        # Final MLP for regression.
        self.fc1 = Lin(4 * dim, dim)
        self.fc2 = Lin(dim, dim)
        self.fc3 = Lin(dim, 1)

    def forward(self, data):
        n = self.var_mlp(data.var_node_features)
        e = self.con_mlp(data.con_node_features)

        x = e.new_zeros((data.node_types.size(0), n.size(-1)))
        x = x.scatter_(0, data.assoc_var.view(-1, 1).expand_as(n), n)
        x = x.scatter_(0, data.assoc_con.view(-1, 1).expand_as(e), e)

        xs = [x]
        xs.append(F.relu(self.conv1(xs[-1], data.edge_index, data.edge_types)))
        xs.append(F.relu(self.conv2(xs[-1], data.edge_index, data.edge_types)))
        xs.append(F.relu(self.conv3(xs[-1], data.edge_index, data.edge_types)))

        x = torch.cat(xs[0:], dim=-1)
        x = x[data.assoc_var]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.fc3(x))

        return x.squeeze(-1)
