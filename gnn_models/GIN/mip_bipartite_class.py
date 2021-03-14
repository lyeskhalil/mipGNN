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

import torch_geometric.utils.softmax

import torch

import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Update constraint embeddings based on variable embeddings.
class VarConBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, var_assigment, aggr):
        super(VarConBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        # Learn joint representation of contraint embedding and error.
        self.joint_var = Sequential(Linear(dim + 1, dim), ReLU(), Linear(dim, dim), ReLU(),
                                    BN(dim))

        # Maps variable embeddings to scalar variable assigment.
        self.var_assigment = var_assigment

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, rhs, size):
        # Compute scalar variable assignment.
        var_assignment = self.var_assigment(source)

        source = self.joint_var(torch.cat([source, var_assignment], dim=-1))

        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)

        tmp = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)

        out = self.mlp((1 + self.eps) * target + tmp)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


# Compute error signal.
class ErrorLayer(MessagePassing):
    def __init__(self, dim, var_assignment):
        super(ErrorLayer, self).__init__(aggr="add", flow="source_to_target")
        self.var_assignment = var_assignment
        self.error_encoder = Sequential(Linear(1, dim), ReLU(), Linear(dim, dim), ReLU(),
                                        BN(dim))

        # Learn joint representation of contraint embedding and error.
        self.joint_var = Sequential(Linear(dim + dim, dim), ReLU(), Linear(dim, dim), ReLU(),
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
        self.joint_var = Sequential(Linear(dim + dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                    BN(dim))

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, error_con, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)

        source = self.joint_var(torch.cat([source, error_con], dim=-1))

        tmp = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)

        out = self.mlp((1 + self.eps) * target + tmp)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden, aggr, num_layers):
        super(SimpleNet, self).__init__()
        self.num_layers = num_layers

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

            x_con.append(F.relu(self.layers_var[-1](x_var[-1], x_con[-1], edge_index_var, edge_features_var, rhs,
                                                    (num_nodes_var.sum(), num_nodes_con.sum()))))

            x_var.append(F.relu(self.layers_con[i](x_con[-1], x_var[-1], edge_index_con, edge_features_con, x_err[-1],
                                                   (num_nodes_con.sum(), num_nodes_var.sum()))))

        x = torch.cat(x_var[:], dim=-1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

# Preprocessing to create Torch dataset.
class GraphDataset(InMemoryDataset):
    def __init__(self, root, data_path, bias_threshold, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_path = data_path
        self.bias_threshold = bias_threshold

    @property
    def raw_file_names(self):
        return sname

    @property
    def processed_file_names(self):
        return sname

    def download(self):
        pass

    def process(self):
        print("Preprocessing.")

        data_list = []
        num_graphs = len(os.listdir(data_path))

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(data_path)):
            print(filename, num, num_graphs)
            if num == 608:
                continue

            # Get graph.
            graph = nx.read_gpickle(data_path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            data = Data()

            #  Maps networkx ids to new variable node ids.
            node_to_varnode = {}
            #  Maps networkx ids to new constraint node ids.
            node_to_connode = {}

            # Number of variables.
            num_nodes_var = 0
            # Number of constraints.
            num_nodes_con = 0
            # Targets (classes).
            y = []
            # Features for variable nodes.
            feat_var = []
            # Feature for constraints nodes.
            feat_con = []
            # Right-hand sides of equations.
            feat_rhs = []

            index = []
            index_var = []
            obj = []

            # Iterate over nodes, and collect features.
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable node.
                if node_data['bipartite'] == 0:
                    node_to_varnode[i] = num_nodes_var
                    num_nodes_var += 1

                    if (node_data['bias'] < bias_threshold):
                        y.append(0)
                    else:
                        y.append(1)

                    feat_var.append([node_data['objcoeff'], graph.degree[i]])
                    obj.append([node_data['objcoeff']])
                    index_var.append(0)

                # Node is constraint node.
                elif node_data['bipartite'] == 1:
                    node_to_connode[i] = num_nodes_con
                    num_nodes_con += 1

                    rhs = node_data['rhs']
                    feat_rhs.append([rhs])
                    feat_con.append([rhs, graph.degree[i]])
                    index.append(0)
                else:
                    print("Error in graph format.")
                    exit(-1)

            # Edge list for var->con graphs.
            edge_list_var = []
            # Edge list for con->var graphs.
            edge_list_con = []

            # Create features matrices for variable nodes.
            edge_features_var = []
            # Create features matrices for constraint nodes.
            edge_features_con = []

            # Remark: graph is directed, i.e., each edge exists for each direction.
            # Flow of messages: source -> target.
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
                # Source node is con, target node is var.

                if graph.nodes[s]['bipartite'] == 1:
                    # Source node is constraint. C->V.
                    edge_list_con.append([node_to_connode[s], node_to_varnode[t]])
                    edge_features_con.append([edge_data['coeff']])
                else:
                    # Source node is variable. V->C.
                    edge_list_var.append([node_to_varnode[s], node_to_connode[t]])
                    edge_features_var.append([edge_data['coeff']])

            edge_index_var = torch.tensor(edge_list_var).t().contiguous()
            edge_index_con = torch.tensor(edge_list_con).t().contiguous()

            # Create data object.
            data.edge_index_var = edge_index_var
            data.edge_index_con = edge_index_con

            data.y = torch.from_numpy(np.array(y)).to(torch.long)
            data.var_node_features = torch.from_numpy(np.array(feat_var)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(feat_con)).to(torch.float)
            data.rhs = torch.from_numpy(np.array(feat_rhs)).to(torch.float)
            data.obj = torch.from_numpy(np.array(obj)).to(torch.float)
            data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
            data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)
            data.num_nodes_var = num_nodes_var
            data.num_nodes_con = num_nodes_con
            data.index = torch.from_numpy(np.array(index)).to(torch.long)
            data.index_var = torch.from_numpy(np.array(index_var)).to(torch.long)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Preprocess indices of bipartite graphs to make batching work.
class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_var']:
            return torch.tensor([self.num_nodes_var, self.num_nodes_con]).view(2, 1)
        elif key in ['edge_index_con']:
            return torch.tensor([self.num_nodes_con, self.num_nodes_var]).view(2, 1)
        elif key in ['index']:
            return torch.tensor(self.num_nodes_con)
        elif key in ['index_var']:
            return torch.tensor(self.num_nodes_var)
        else:
            return 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


print(sys.argv[1])
i = int(sys.argv[1])

file_list = [
    "../../DATA1/er_SET2/200_200/alpha_0.75_setParam_100/train/",
    "../../DATA1/er_SET2/200_200/alpha_0.25_setParam_100/train/",
    "../../DATA1/er_SET2/200_200/alpha_0.5_setParam_100/train/",
    "../../DATA1/er_SET2/300_300/alpha_0.75_setParam_100/train/",
    "../../DATA1/er_SET2/300_300/alpha_0.25_setParam_100/train/",
    "../../DATA1/er_SET2/300_300/alpha_0.5_setParam_100/train/",
    "../../DATA1/er_SET1/400_400/alpha_0.75_setParam_100/train/",
    "../../DATA1/er_SET1/400_400/alpha_0.5_setParam_100/train/",
    # "../../DATA1/er_SET1/400_400/alpha_0.25_setParam_100/train/",
]

name_list = [
    "er_SET2_200_200_alpha_0_75_setParam_100_train",
    "er_SET2_200_200_alpha_0_25_setParam_100_train",
    "er_SET2_200_200_alpha_0_5_setParam_100_train",
    "er_SET2_300_300_alpha_0_75_setParam_100_train",
    "er_SET2_300_300_alpha_0_25_setParam_100_train",
    "er_SET2_300_300_alpha_0_5_setParam_100_train",
    "er_SET1_400_400_alpha_0_75_setParam_100_train",
    "er_SET1_400_400_alpha_0_5_setParam_100_train",
    # "er_SET1_400_400_alpha_0_25_setParam_100_train",
]

print(name_list[i])

path = file_list[i]
name = name_list[i]

results = []

# Prepare data.
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'DS')
# Path to raw graph data.
data_path = path
sname = name
# Threshold for computing class labels.
bias_threshold = 0.050
# Create dataset.
dataset = GraphDataset(path, data_path, bias_threshold, transform=MyTransform())  # .shuffle()

# Split data.
l = len(dataset)
train_index, rest = train_test_split(list(range(0, l)), test_size=0.2)
l = len(rest)
val_index = rest[0:int(l / 2)]
test_index = rest[int(l / 2):]

train_dataset = dataset[train_index].shuffle()
val_dataset = dataset[val_index].shuffle()
test_dataset = dataset[test_index].shuffle()

batch_size = 15
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    l = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        pred = pred.max(dim=1)[1]
        correct += pred.eq(data.y).float().mean().item()
        l += 1

    return correct / l


best_val = 0.0
test_acc = 0.0
best_hp = []

plots = []

for i in range(5):
    p = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(hidden=128, num_layers=4, aggr="add").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.8, patience=10,
                                                           min_lr=0.0000001)
    for epoch in range(1, 50):
        print(i)

        train_loss = train(epoch)
        train_acc = test(train_loader)

        val_acc = test(val_loader)
        scheduler.step(val_acc)
        lr = scheduler.optimizer.param_groups[0]['lr']

        if val_acc > best_val:
            best_val = val_acc
            test_acc = test(test_loader)

        # Break if learning rate is smaller 10**-6.
        if lr < 0.000001:
            results.append(test_acc)
            break

        print('Epoch: {:03d}, LR: {:.7f}, Train Loss: {:.7f},  '
              'Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, lr, train_loss,
                                                                            train_acc, val_acc, test_acc))

        p.append(test_acc)
    plots.append(p)

print(plots)
