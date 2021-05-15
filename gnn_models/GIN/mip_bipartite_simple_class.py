import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr):
        super(SimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)

        tmp = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)

        out = self.mlp((1 + self.eps) * target + tmp)

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
        self.var_node_encoder = Sequential(Linear(1, hidden), ReLU(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(Linear(1, hidden), ReLU(), Linear(hidden, hidden))

        # Bipartite GNN architecture.
        self.layers_con = []
        self.layers_var = []
        for i in range(self.num_layers):
            self.layers_con.append(SimpleBipartiteLayer(1, hidden, aggr=aggr))
            self.layers_var.append(SimpleBipartiteLayer(1, hidden, aggr=aggr))

        self.layers_con = torch.nn.ModuleList(self.layers_con)
        self.layers_var = torch.nn.ModuleList(self.layers_var)

        # MLP used for classification.
        self.lin1 = Linear((num_layers + 1) * hidden, hidden)
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

        # Compute initial node embeddings.
        var_node_features_0 = self.var_node_encoder(var_node_features)
        con_node_features_0 = self.con_node_encoder(con_node_features)

        x_var = [var_node_features_0]
        x_con = [con_node_features_0]

        for i in range(self.num_layers):
            x_con.append(F.relu(self.layers_var[i](x_var[-1], x_con[-1], edge_index_var, edge_features_var,
                                                   (var_node_features_0.size(0), con_node_features.size(0)))))

            x_var.append(F.relu(self.layers_con[i](x_con[-1], x_var[-1], edge_index_con, edge_features_con,
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
