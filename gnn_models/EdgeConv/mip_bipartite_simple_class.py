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

        self.nn = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                             BN(dim))

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

    def forward(self, source, target, edge_index, edge_attr, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)

        out = self.propagate(edge_index, x=source, t=target, edge_attr=edge_embedding, size=size)

        return out

    def message(self, x_j, t_i, edge_attr):
        return self.nn(torch.cat([t_i, x_j, edge_attr], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden, aggr, num_layers):
        super(SimpleNet, self).__init__()
        self.num_layers = num_layers

        # Embed initial node features.
        self.var_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))

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

        num_var = var_node_features_0.size(0)
        num_con = con_node_features_0.size(0)

        for i in range(self.num_layers):
            x_con.append(F.relu(self.layers_var[i](x_var[-1], x_con[-1], edge_index_var, edge_features_var,
                                                   (num_var, num_con))))

            x_var.append(F.relu(self.layers_con[i](x_con[-1], x_var[-1], edge_index_con, edge_features_con,
                                                   (num_con, num_var))))

        x = torch.cat(x_var[:], dim=-1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1)  # , F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
