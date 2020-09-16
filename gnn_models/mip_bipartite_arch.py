import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

from torch_scatter import scatter_add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Update constraint embeddings based on variable embeddings.
class VarConBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, var_assigment):
        super(VarConBipartiteLayer, self).__init__(aggr="add", flow="source_to_target")

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        # Maps variable embeddings to scalar variable assigment.
        self.var_assigment = var_assigment
        # Maps variable embeddings + assignment to joint embedding.
        self.joint_var = Sequential(Linear(dim + 1, dim), ReLU(), Linear(dim, dim), ReLU(),
                                    BN(dim))

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                              BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, rhs, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)
        # Compute scalar variable assignment.
        var_assignment = self.var_assigment(source)
        # Compute joint embedding of variable embeddings and scalar variable assignment.
        new_source = self.joint_var(torch.cat([source, var_assignment], dim=-1))

        # Do the acutal message passing.
        tmp = self.propagate(edge_index, x=new_source, var_assignment=var_assignment, edge_attr=edge_embedding,
                             size=size)
        out = self.mlp((1 + self.eps) * target + tmp)

        return out

    def message(self, x_j, var_assignment_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset(self.edge_encoder)
        reset(self.var_assigment)
        reset(self.node_encoder)
        reset(self.joint_var)
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)


# Compute error signal.
class ErrorLayer(MessagePassing):
    def __init__(self, dim, var_assignment):
        super(ErrorLayer, self).__init__(aggr="add", flow="source_to_target")
        self.var_assignment = var_assignment
        self.error_encoder = Sequential(Linear(1, dim), ReLU(), Linear(dim, dim), ReLU(),
                                        BN(dim))

    # TODO: Change back!
    def forward(self, source, edge_index, edge_attr, rhs, index, size):
        # Compute scalar variable assignment.
        new_source = self.var_assignment(source)
        tmp = self.propagate(edge_index, x=new_source, edge_attr=edge_attr, size=size)

        # Compute residual, i.e., Ax-b.
        out = tmp - rhs

        # TODO: Think here.
        #out = self.error_encoder(out)

        # TODO: Change.
        #out = softmax(out, index)

        return out

    def message(self, x_j, edge_attr):
        msg = x_j * edge_attr

        return msg

    def update(self, aggr_out):
        return aggr_out


# Update variable embeddings based on constraint embeddings.
class ConVarBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim):
        super(ConVarBipartiteLayer, self).__init__(aggr="add", flow="source_to_target")

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        # Learn joint representation of contraint embedding and error.
        self.joint_con_encoder = Sequential(Linear(dim + 1, dim), ReLU(), Linear(dim, dim-1), ReLU(),
                                            BN(dim-1))

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, error_con, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)

        joint_con = torch.cat([self.joint_con_encoder(torch.cat([source, error_con], dim=-1)), error_con], dim=-1)
        #joint_con = self.joint_con_encoder(torch.cat([source, error_con], dim=-1))
        tmp = self.propagate(edge_index, x=joint_con, error=error_con, edge_attr=edge_embedding, size=size)

        out = self.mlp((1 + self.eps) * target + tmp)

        return out

    def message(self, x_j, error_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset(self.node_encoder)
        reset(self.edge_encoder)
        reset(self.joint_con_encoder)
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden):
        super(SimpleNet, self).__init__()

        # Embed initial node features.
        self.var_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))

        # Compute variable assignement.
        # TODO: Just one shared assignment for all layers?
        self.var_assigment_1 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1), Sigmoid())
        self.var_assigment_2 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1), Sigmoid())
        self.var_assigment_3 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1), Sigmoid())
        self.var_assigment_4 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1), Sigmoid())

        # Bipartite GNN architecture.
        self.var_con_1 = VarConBipartiteLayer(1, hidden, self.var_assigment_1)
        self.error_1 = ErrorLayer(hidden, self.var_assigment_1)

        self.con_var_1 = ConVarBipartiteLayer(1, hidden)
        self.var_con_2 = VarConBipartiteLayer(1, hidden, self.var_assigment_2)
        self.error_2 = ErrorLayer(hidden, self.var_assigment_2)

        self.con_var_2 = ConVarBipartiteLayer(1, hidden)
        self.var_con_3 = VarConBipartiteLayer(1, hidden, self.var_assigment_3)
        self.error_3 = ErrorLayer(hidden, self.var_assigment_3)

        self.con_var_3 = ConVarBipartiteLayer(1, hidden)
        self.var_con_4 = VarConBipartiteLayer(1, hidden, self.var_assigment_4)
        self.error_4 = ErrorLayer(hidden, self.var_assigment_4)

        self.con_var_4 = ConVarBipartiteLayer(1, hidden)

        # MLP used for classification.
        self.lin1 = Linear(5 * hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.lin4 = Linear(hidden, 2)

    def reset_parameters(self):
        self.var_node_encoder.reset_parameters()
        self.con_node_encoder.reset_parameters()

        self.var_assigment_1.reset_parameters()
        self.var_assigment_2.reset_parameters()
        self.var_assigment_3.reset_parameters()
        self.var_assigment_4.reset_parameters()

        self.var_con_1.reset_parameters()
        self.con_var_1.reset_parameters()

        self.var_con_2.reset_parameters()
        self.con_var_2.reset_parameters()

        self.var_con_3.reset_parameters()
        self.con_var_3.reset_parameters()

        self.var_con_4.reset_parameters()
        self.con_var_4.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, data):
        # Get data of batch.
        var_node_features = data.var_node_features
        con_node_features = data.con_node_features
        edge_index_var = data.edge_index_var
        edge_index_con = data.edge_index_con
        edge_features_var = data.edge_features_var
        edge_features_con = data.edge_features_con
        num_nodes_var = data.num_nodes_var
        num_nodes_con = data.num_nodes_con
        rhs = data.rhs
        index = data.index
        obj = data.obj

        # Compute initial node embeddings.
        var_node_features_0 = self.var_node_encoder(var_node_features)
        con_node_features_0 = self.con_node_encoder(con_node_features)


        con_node_features_1 = F.relu(
            self.var_con_1(var_node_features_0, con_node_features_0, edge_index_var, edge_features_var, rhs,
                           (var_node_features_0.size(0), con_node_features.size(0))))
        err_1 = self.error_1(var_node_features_0, edge_index_var, edge_features_var, rhs, index,
                             (var_node_features_0.size(0), con_node_features.size(0)))

        var_node_features_1 = F.relu(
            self.con_var_1(con_node_features_1, var_node_features_0, edge_index_con, edge_features_con, err_1,
                           (con_node_features_1.size(0), var_node_features_0.size(0))))

        con_node_features_2 = F.relu(
            self.var_con_2(var_node_features_1, con_node_features_1, edge_index_var, edge_features_var, rhs,
                           (var_node_features_1.size(0), con_node_features_1.size(0))))
        err_2 = self.error_1(var_node_features_1, edge_index_var, edge_features_var, rhs, index,
                             (var_node_features_1.size(0), con_node_features_1.size(0)))

        var_node_features_2 = F.relu(
            self.con_var_2(con_node_features_2, var_node_features_1, edge_index_con, edge_features_con, err_2,
                           (con_node_features_2.size(0), var_node_features_1.size(0))))

        con_node_features_3 = F.relu(
            self.var_con_3(var_node_features_2, con_node_features_2, edge_index_var, edge_features_var, rhs,
                           (var_node_features_2.size(0), con_node_features_2.size(0))))
        err_3 = self.error_1(var_node_features_2, edge_index_var, edge_features_var, rhs, index,
                             (var_node_features_2.size(0), con_node_features_2.size(0)))

        var_node_features_3 = F.relu(
            self.con_var_3(con_node_features_3, var_node_features_2, edge_index_con, edge_features_con, err_3,
                           (con_node_features_3.size(0), var_node_features_2.size(0))))

        con_node_features_4 = F.relu(
            self.var_con_4(var_node_features_3, con_node_features_3, edge_index_var, edge_features_var, rhs,
                           (var_node_features_3.size(0), con_node_features_3.size(0))))
        err_4 = self.error_1(var_node_features_3, edge_index_var, edge_features_var, rhs, index,
                             (var_node_features_3.size(0), con_node_features_3.size(0)))

        var_node_features_4 = F.relu(
            self.con_var_4(con_node_features_4, var_node_features_3, edge_index_con, edge_features_con, err_4,
                           (con_node_features_4.size(0), var_node_features_3.size(0))))

        var = self.var_assigment_4(var_node_features_4)

        # cost = torch.mul(var, obj)
        # print(cost.size())
        # print(data.index_var)
        # cost = scatter_add(cost, index=data.index_var, dim=0)
        #
        # print(cost.size())
        # exit()


        # print(err_1.min(), print(err_1.max()))

        x = torch.cat([var_node_features_0, var_node_features_1, var_node_features_2, var_node_features_3, var_node_features_4], dim=-1)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin3(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
