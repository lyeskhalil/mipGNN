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
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

from torchmetrics import F1, Precision, Recall, Accuracy

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
                                                   (num_nodes_var.sum(), num_nodes_con.sum()))))

            x_var.append(F.relu(self.layers_con[i](x_con[-1], x_var[-1], edge_index_con, edge_features_con,
                                                   (num_nodes_con.sum(), num_nodes_var.sum()))))

        x = torch.cat(x_var[:], dim=-1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1), F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# Preprocessing to create Torch dataset.
class GraphDataset(InMemoryDataset):

    def __init__(self, name, root, data_path, bias_threshold, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.bias_threshold = bias_threshold
        global global_name
        global global_data_path

    @property
    def raw_file_names(self):
        return name

    @property
    def processed_file_names(self):
        return name

    def download(self):
        pass

    def process(self):
        print("Preprocessing.")

        data_list = []
        num_graphs = len(os.listdir(pd))

        print(pd)

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(pd)):
            print(filename, num, num_graphs)

            # Get graph.
            graph = nx.read_gpickle(pd + filename)

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
            y_real = []
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

                    y_real.append(node_data['bias'])
                    if (node_data['bias'] < bias_threshold):
                        y.append(0)
                    else:
                        y.append(1)

                    if 'objcoeff' in node_data:
                        feat_var.append([node_data['objcoeff'], graph.degree[i]])
                        obj.append([node_data['objcoeff']])
                    else:
                        feat_var.append([node_data['obj_coeff'], graph.degree[i]])
                        obj.append([node_data['obj_coeff']])

                    index_var.append(0)

                # Node is constraint node.
                elif node_data['bipartite'] == 1:
                    node_to_connode[i] = num_nodes_con
                    num_nodes_con += 1

                    if 'rhs' in node_data:
                        rhs = node_data['rhs']
                    else:
                        rhs = node_data['bound']

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
            data.y_real = torch.from_numpy(np.array(y_real)).to(torch.float)
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


dataset_list = [
    "../../data_new/data_graphsonly/gisp/p_hat300-2.clq/train/",
    "../../data_new/data_graphsonly/gisp/p_hat300-2.clq/test/",
    "../../data_new/data_graphsonly/gisp/C250.9.clq/train/",
    "../../data_new/data_graphsonly/gisp/C250.9.clq/test/",
    "../../data_new/data_graphsonly/gisp/keller4.clq/train/",
    "../../data_new/data_graphsonly/gisp/keller4.clq/test/",
    "../../data_new/data_graphsonly/gisp/hamming8-4.clq/train/",
    "../../data_new/data_graphsonly/gisp/hamming8-4.clq/test/",
    "../../data_new/data_graphsonly/gisp/gen200_p0.9_55.clq/train/",
    "../../data_new/data_graphsonly/gisp/gen200_p0.9_55.clq/test/",
    "../../data_new/data_graphsonly/gisp/gen200_p0.9_44.clq/train/",
    "../../data_new/data_graphsonly/gisp/gen200_p0.9_44.clq/test/",
    "../../data_new/data_graphsonly/gisp/C125.9.clq/train/",
    "../../data_new/data_graphsonly/gisp/C125.9.clq/test/",
    "../../data_new/data_graphsonly/gisp/p_hat300-1.clq/train/",
    "../../data_new/data_graphsonly/gisp/p_hat300-1.clq/test/",
    "../../data_new/data_graphsonly/gisp/brock200_4.clq/train/",
    "../../data_new/data_graphsonly/gisp/brock200_4.clq/test/",
    "../../data_new/data_graphsonly/gisp/brock200_2.clq/train/",
    "../../data_new/data_graphsonly/gisp/brock200_2.clq/test/",
    "../../data_new/data_graphsonly/fcmnf/L_n200_p0.02_c500/train/",
    "../../data_new/data_graphsonly/fcmnf/L_n200_p0.02_c500/test/"
]

name_list = [
    "1_test",
    "1_train",
    "2_test",
    "2_train",
    "3_test",
    "3_train",
    "4_test",
    "4_train",
    "5_test",
    "5_train",
    "6_test",
    "6_train",
    "7_test",
    "7_train",
    "8_test",
    "8_train",
    "9_test",
    "9_train",
    "10_test",
    "10_train",
    "11_test",
    "11_train",
]

pathr = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'DS')
bias_threshold = 0.001

results = []

i = 4
for _ in range(4):
    pd = path_train = path_trainpath_train = dataset_list[i]
    name = name_train = name_list[i]
    train_dataset = GraphDataset(name_train, pathr, path_train, bias_threshold, transform=MyTransform()).shuffle()

    pd = path_test = path_testpath_test = dataset_list[i + 1]
    name = name_test = name_list[i+1]
    test_dataset = GraphDataset(name_test, pathr, path_test, bias_threshold, transform=MyTransform()).shuffle()

    print("###")
    zero = torch.tensor([0])
    one = torch.tensor([1])
    print(torch.where(test_dataset.data.y_real <= bias_threshold, zero, one).to(torch.float).mean())

    # Split data.
    l = len(train_dataset)
    train_index, val_index = train_test_split(list(range(0, l)), test_size=0.2)
    l = len(val_index)

    val_dataset = train_dataset[val_index].shuffle()
    train_dataset = train_dataset[train_index].shuffle()
    test_dataset = test_dataset.shuffle()  # [0:200]

    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    def train(epoch):
        model.train()

        loss_all = 0
        zero = torch.tensor([0]).to(device)
        one = torch.tensor([1]).to(device)

        # TODO
        # weights = torch.tensor([0.05, 0.95]).to(device)
        for data in train_loader:
            data = data.to(device)

            y = data.y_real
            y = torch.where(y <= bias_threshold, zero, one).to(device)

            optimizer.zero_grad()
            output, softmax = model(data)

            # TODO
            loss = F.nll_loss(output, y)
            loss.backward()
            loss_all += batch_size * loss.item()
            optimizer.step()

        return loss_all / len(train_dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        correct = 0
        l = 0

        zero = torch.tensor([0]).to(device)
        one = torch.tensor([1]).to(device)
        f1 = F1(num_classes=2, average="macro").to(device)
        pr = Precision(num_classes=2, average="macro").to(device)
        re = Recall(num_classes=2, average="macro").to(device)
        acc = Accuracy(num_classes=2).to(device)

        for data in loader:
            data = data.to(device)
            pred, softmax = model(data)

            y = data.y_real
            y = torch.where(y <= bias_threshold, zero, one).to(device)
            pred = pred.max(dim=1)[1]

            if l == 1:
                pred_all = torch.cat([pred_all, pred])
                y_all = torch.cat([y_all, y])
            else:
                pred_all = pred
                y_all = y

            correct += pred.eq(y).float().mean().item()
            l += 1

        return acc(pred_all, y_all), f1(pred_all, y_all), pr(pred_all, y_all), re(pred_all, y_all)


    best_val = 0.0
    test_acc = 0.0
    test_f1 = 0.0
    test_re = 0.0
    test_pr = 0.0
    r = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(hidden=64, num_layers=5, aggr="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.8, patience=5,
                                                           min_lr=0.0000001)

    for epoch in range(1, 50):
        print(i)

        train_loss = train(epoch)
        train_acc, train_f1, train_pr, train_re = test(train_loader)

        val_acc, val_f1, val_pr, val_re = test(val_loader)
        scheduler.step(val_acc)
        lr = scheduler.optimizer.param_groups[0]['lr']

        if val_acc > best_val:
            best_val = val_acc
            test_acc, test_f1, test_pr, test_re = test(test_loader)

        r.append(test_acc)

        # Break if learning rate is smaller 10**-6.
        if lr < 0.000001:
            break

        print('Epoch: {:03d}, LR: {:.7f}, Train Loss: {:.7f},  '
              'Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, lr, train_loss,
                                                                            train_acc, val_acc, test_acc))

        print("F1", train_f1, val_f1, test_f1)
        print("Pr", train_pr, val_pr, test_pr)
        print("Re", train_re, val_re, test_re)

    results.append(r)
    i += 2

print(results)
