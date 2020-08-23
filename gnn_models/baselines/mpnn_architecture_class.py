import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.nn import NNConv




class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))
        self.con_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))

        nn1 = Seq(Lin(2, dim), ReLU(), Lin(dim, dim * dim))
        self.conv1 = NNConv(dim, dim, nn1, aggr='mean', root_weight=True)

        nn2 = Seq(Lin(2, dim), ReLU(), Lin(dim, dim * dim))
        self.conv2 = NNConv(dim, dim, nn2, aggr='mean', root_weight=True)

        nn3 = Seq(Lin(2, dim), ReLU(), Lin(dim, dim * dim))
        self.conv3 = NNConv(dim, dim, nn3, aggr='mean', root_weight=True)

        nn4 = Seq(Lin(2, dim), ReLU(), Lin(dim, dim * dim))
        self.conv4 = NNConv(dim, dim, nn4, aggr='mean', root_weight=True)

        # Final MLP for regression.
        self.fc1 = Lin(4 * dim, dim)
        self.fc2 = Lin(dim, dim)
        self.fc3 = Lin(dim, dim)
        # self.fc4 = Lin(dim, dim)
        # self.fc5 = Lin(dim, dim)
        self.fc6 = Lin(dim, 2)

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
        xs.append(F.relu(self.conv4(xs[-1], data.edge_index, data.edge_types)))

        x = torch.cat(xs, dim=-1)
        x = x[data.assoc_var]

        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x