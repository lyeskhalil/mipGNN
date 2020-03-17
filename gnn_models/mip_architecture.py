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

        self.w_cons = Param(torch.Tensor(in_channels, out_channels))
        self.w_var = Param(torch.Tensor(in_channels, out_channels))

        self.root = Param(torch.Tensor(in_channels, out_channels))
        self.bias = Param(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.w_cons)
        uniform(size, self.w_cons)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_feature, size=None):
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_feature=edge_feature)

    def message(self, x_j, edge_index_j, edge_type, edge_feature):
        # Split data.
        # x_j_0 = x_j[edge_type == 0]
        # x_j_1 = x_j[edge_type == 1]
        #
        # # TODO: Check direction of message.
        # c = edge_feature[edge_index_j][edge_type == 1]
        # var_assign = self.hidden_to_var(x_j_1)
        # var_assign = var_assign * c

        out_0 = torch.matmul(x_j, self.w_cons)
        # out_1 = torch.matmul(x_j_1, self.w_var)
        # zeros = torch.zeros(out_0.size(0), 1,device=torch.device("cuda"))
        #
        # new_out = torch.Tensor(edge_type.size(0), self.out_channels).cuda()
        #
        # new_out[edge_type == 0] = out_0
        # new_out[edge_type == 1] = out_1



        return out_0

    def update(self, aggr_out, x):
        # Compute violation of constraint.
        #aggr_out[:, -1] = aggr_out[:,-1] - x[:,-1]
        # aggr_out = self.proj(aggr_out)

        aggr_out = aggr_out + torch.matmul(x, self.root)
        aggr_out = aggr_out + self.bias

        return aggr_out


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(1, dim-1), ReLU(), Lin(dim-1, dim-1))
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
        n = torch.cat([self.var_mlp(data.var_node_features),data.var_node_features], dim=-1)
        e = torch.cat([self.con_mlp(data.con_node_features),data.con_node_features], dim=-1)

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

