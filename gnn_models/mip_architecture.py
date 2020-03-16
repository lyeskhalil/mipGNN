import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx



from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import NNConv

import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform


class MIPGNN(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(MIPGNN, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_feature, size=None):
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_feature=edge_feature)

    def message(self, x_j, edge_index_j, edge_type, edge_feature):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))
        self.con_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))

        self.conv1 = MIPGNN(dim, dim, 2, root_weight=True, bias=True, num_bases=5)
        self.conv2 = MIPGNN(dim, dim, 2, root_weight=True, bias=True, num_bases=5)
        self.conv3 = MIPGNN(dim, dim, 2, root_weight=True, bias=True, num_bases=5)
        self.conv4 = MIPGNN(dim, dim, 2, root_weight=True, bias=True, num_bases=5)

        # Final MLP for regression.
        self.fc1 = Lin(5 * dim, dim)
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
        xs.append(F.relu(self.conv1(xs[-1], data.edge_index, data.edge_types, data.edge_features)))
        xs.append(F.relu(self.conv2(xs[-1], data.edge_index, data.edge_types, data.edge_features)))
        xs.append(F.relu(self.conv3(xs[-1], data.edge_index, data.edge_types, data.edge_features)))
        xs.append(F.relu(self.conv4(xs[-1], data.edge_index, data.edge_types, data.edge_features)))

        x = torch.cat(xs[0:], dim=-1)
        x = x[data.assoc_var]

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc4(x)

        return x.squeeze(-1)

