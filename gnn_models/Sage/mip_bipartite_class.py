import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import torch_geometric.utils.softmax
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Update constraint embeddings based on variable embeddings.
class VarConBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, var_assigment, aggr):
        super(VarConBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        self.lin_l = Linear(dim, dim, bias=True)
        self.lin_r = Linear(dim, dim, bias=False)

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

        out = self.propagate(edge_index, x=new_source, size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)

        out += self.lin_r(target)

        out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# Compute error signal.
class ErrorLayer(MessagePassing):
    def __init__(self, dim, var_assignment):
        super(ErrorLayer, self).__init__(aggr="add", flow="source_to_target")
        self.var_assignment = var_assignment
        self.error_encoder = Sequential(Linear(1, dim), ReLU(), Linear(dim, dim), ReLU(),
                                        BN(dim))

    def forward(self, source, edge_index, edge_attr, rhs, index, size):
        # Compute scalar variable assignment.
        new_source = self.var_assignment(source)
        tmp = self.propagate(edge_index, x=new_source, edge_attr=edge_attr, size=size)

        # Compute residual, i.e., Ax-b.
        out = tmp - rhs

        out = self.error_encoder(out)

        out = torch_geometric.utils.softmax(out, index)

        return out

    def message(self, x_j, edge_attr):
        msg = x_j * edge_attr

        return msg

    def update(self, aggr_out):
        return aggr_out


# Update variable embeddings based on constraint embeddings.
class ConVarBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr):
        super(ConVarBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        # Learn joint representation of contraint embedding and error.
        self.joint_con_encoder = Sequential(Linear(dim + dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                            BN(dim))

        self.lin_l = Linear(dim, dim, bias=True)
        self.lin_r = Linear(dim, dim, bias=False)

    def forward(self, source, target, edge_index, edge_attr, error_con, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)
        new_source = self.joint_con_encoder(torch.cat([source, error_con], dim=-1))

        out = self.propagate(edge_index, x=new_source, size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)

        out += self.lin_r(target)
        out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden, aggr, num_layers, regression):
        super(SimpleNet, self).__init__()
        self.num_layers = num_layers

        self.regression = regression

        # Embed initial node features.
        self.var_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))

        # Compute variable assignement.
        self.layers_ass = []
        for i in range(self.num_layers):
            self.layers_ass.append(Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1), Sigmoid()))

        # Bipartite GNN architecture.
        self.layers_con = []
        self.layers_var = []
        self.layers_err = []

        for i in range(self.num_layers):
            self.layers_con.append(ConVarBipartiteLayer(1, hidden, aggr=aggr))
            self.layers_var.append(VarConBipartiteLayer(1, hidden, self.layers_ass[i], aggr=aggr))
            self.layers_err.append(ErrorLayer(hidden, self.layers_ass[i]))

        self.layers_con = torch.nn.ModuleList(self.layers_con)
        self.layers_var = torch.nn.ModuleList(self.layers_var)
        self.layers_err = torch.nn.ModuleList(self.layers_err)

        # MLP used for classification.
        self.lin1 = Linear((self.num_layers + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.lin4 = Linear(hidden, 2)

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

        x_var = [var_node_features_0]
        x_con = [con_node_features_0]
        x_err = []

        for i in range(self.num_layers):
            x_err.append(self.layers_err[i](x_var[-1], edge_index_var, edge_features_var, rhs, index,
                                            (var_node_features_0.size(0), con_node_features.size(0))))

            x_con.append(F.relu(self.layers_var[i](x_var[-1], x_con[-1], edge_index_var, edge_features_var, rhs,
                                                    (var_node_features_0.size(0), con_node_features.size(0)))))

            x_var.append(F.relu(self.layers_con[i](x_con[-1], x_var[-1], edge_index_con, edge_features_con, x_err[-1],
                                                   (con_node_features.size(0), var_node_features_0.size(0)))))


        x = torch.cat(x_var[:], dim=-1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)

        if not self.regression:
            return F.log_softmax(x, dim=-1)
        else:
            return x.view(-1)

    def __repr__(self):
        return self.__class__.__name__

