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
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.nn import NNConv

import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform


class MIPGNN(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(MIPGNN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hidden_to_var = Seq(Lin(in_channels, in_channels), Sigmoid(), Lin(in_channels, 1))
        self.proj = Seq(Lin(in_channels+1, out_channels), ReLU(), Lin(out_channels, out_channels))

        self.w_cons = Param(torch.Tensor(in_channels-2, out_channels-2))
        self.w_vars = Param(torch.Tensor(in_channels, out_channels))

        self.root_cons = Param(torch.Tensor(in_channels, out_channels))
        self.root_vars = Param(torch.Tensor(in_channels, out_channels))
        self.bias = Param(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size-2, self.w_cons)
        uniform(size, self.w_vars)
        uniform(size, self.root_cons)
        uniform(size, self.root_vars)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_feature, assoc_con, assoc_var, size=None):
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_feature=edge_feature, assoc_con=assoc_con, assoc_var=assoc_var)

    def message(self, x_j, edge_index_j, edge_type, edge_feature):
        # Split data.
        # Variable nodes.
        x_j_0 = x_j[edge_type == 0]
        # Constraint nodes.
        x_j_1 = x_j[edge_type == 1]


        ### Vars -> Cons.
        #  x_j is variable nodes.
        c = edge_feature[edge_index_j][edge_type == 0]
        var_assign = self.hidden_to_var(x_j_0)
        # Variable assignment * coeffient in contraint.
        var_assign = var_assign * c
        out_0 = torch.matmul(x_j_0[:, 0:-2], self.w_cons)
        # Assign left side of constraint to last column.
        out_0 = torch.cat([out_0, x_j_0[:, -2].view(x_j_0.size(0), 1), var_assign], dim=-1)

        ### Cons -> Vars.
        out_1 = torch.matmul(x_j_1, self.w_vars)
        new_out = torch.zeros(x_j.size(0), self.out_channels, device=torch.device("cuda"))

        new_out[edge_type == 0] = out_0
        new_out[edge_type == 1] = out_1


        return new_out

    def update(self, aggr_out, x, assoc_con, assoc_var):

        # Compute violation of constraint.
        # t = aggr_out[assoc_con,-1] - x[assoc_con,-1]
        # aggr_out[assoc_con, -1] = t



        t_1 = aggr_out[assoc_var] + torch.matmul(x[assoc_var], self.root_vars)
        t_2 = aggr_out[assoc_con] + torch.matmul(x[assoc_con], self.root_vars)
        new_out = torch.zeros(aggr_out.size(0), aggr_out.size(1), device=torch.device("cuda"))
        new_out[assoc_var] = t_1
        new_out[assoc_con] = t_2

        aggr_out = aggr_out + self.bias

        return aggr_out


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(1, dim-2), ReLU(), Lin(dim-2, dim-2))
        self.con_mlp = Seq(Lin(1, dim-1), ReLU(), Lin(dim-1, dim-1))

        self.conv1 = MIPGNN(dim, dim)
        self.conv2 = MIPGNN(dim, dim)
        self.conv3 = MIPGNN(dim, dim)
        self.conv4 = MIPGNN(dim, dim)

        # Final MLP for regression.
        self.fc1 = Lin(5 * dim, dim)
        self.fc2 = Lin(dim, dim)
        self.fc3 = Lin(dim, dim)
        self.fc4 = Lin(dim, 1)

    def forward(self, data):

        ones = torch.ones(data.var_node_features.size(0), 1).cuda()
        n = torch.cat([self.var_mlp(data.var_node_features),data.var_node_features,ones], dim=-1)
        e = torch.cat([self.con_mlp(data.con_node_features),data.con_node_features], dim=-1)

        x = e.new_zeros((data.node_types.size(0), n.size(-1)))
        x = x.scatter_(0, data.assoc_var.view(-1, 1).expand_as(n), n)
        x = x.scatter_(0, data.assoc_con.view(-1, 1).expand_as(e), e)

        xs = [x]
        xs.append(F.relu(self.conv1(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var)))
        xs.append(F.relu(self.conv2(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var)))
        xs.append(F.relu(self.conv3(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var)))
        xs.append(F.relu(self.conv4(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var)))


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

