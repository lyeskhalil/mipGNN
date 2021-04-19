import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import networkx as nx

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np

import torch

import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class SimpleBipartiteLayer(MessagePassing):
    def __init__(self, dim, aggr):
        super(SimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, edge_index):
        tmp = self.propagate(edge_index, x=source)

        out = self.mlp((1 + self.eps) * source + tmp)

        return out

    def message(self, x_j):
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden, aggr, ):
        super(SimpleNet, self).__init__()

        # Embed initial node features.
        self.node_encoder = Sequential(Linear(8, hidden), ReLU(), Linear(hidden, hidden))

        self.conv_1_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.conv_2_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.joint_1 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        self.conv_1_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.conv_2_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.joint_2 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        self.conv_1_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.conv_2_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.joint_3 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        self.conv_1_4 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.conv_2_4 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.joint_4 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        # MLP used for classification.
        self.lin1 = Linear(5 * hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.lin4 = Linear(hidden, 2)

    def forward(self, data):
        # Get data of batch.
        node_features_0 = data.node_features

        print(node_features_0.size())
        exit()
        edge_index_1 = data.edge_index_1
        edge_index_2 = data.edge_index_2
        indices = data.indices

        # Compute initial node embeddings.
        node_features_0 = self.node_encoder(node_features_0)

        x_1 = self.conv_1_1(node_features_0, edge_index_1)
        x_2 = self.conv_2_1(node_features_0, edge_index_2)
        node_features_1 = self.joint_1(torch.cat([x_1, x_2] , dim=-1))

        x_1 = self.conv_1_2(node_features_1, edge_index_1)
        x_2 = self.conv_2_2(node_features_1, edge_index_2)
        node_features_2 = self.joint_1(torch.cat([x_1, x_2] , dim=-1))

        x_1 = self.conv_1_3(node_features_2, edge_index_1)
        x_2 = self.conv_2_3(node_features_2, edge_index_2)
        node_features_3 = self.joint_1(torch.cat([x_1, x_2] , dim=-1))

        x_1 = self.conv_1_4(node_features_3, edge_index_1)
        x_2 = self.conv_2_4(node_features_3, edge_index_2)
        node_features_4 = self.joint_1(torch.cat([x_1, x_2] , dim=-1))

        x = torch.cat([node_features_0, node_features_1, node_features_2, node_features_3, node_features_4], dim=-1)[indices]

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = "../../DATA1/er_SET2/200_200/alpha_0.75_setParam_100/train/"


# Preprocessing to create Torch dataset.
class GraphDataset(InMemoryDataset):
    def __init__(self, root, bias_threshold, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.bias_threshold = bias_threshold

    @property
    def raw_file_names(self):
        return "tedsfsdst"

    @property
    def processed_file_names(self):
        return "tfffest"

    def download(self):
        pass

    def process(self):
        print("Preprocessing.")

        data_list = []

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(data_path)):
            print(num)
            # Get graph.
            graph = nx.read_gpickle(data_path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph

            graph_new = nx.Graph()

            matrices_1 = []
            matrices_2 = []
            features = []
            indices = []
            y = []
            num = 0

            for i, u in enumerate(graph.nodes):
                for j, v in enumerate(graph.nodes):
                    if graph.has_edge(u,v):
                        if graph.nodes[u]['bipartite'] == 0 and graph.nodes[v]['bipartite'] == 0:
                            graph_new.add_node((u, v), type="VV", first = u, second = v, num=num)

                            features.append([graph.nodes[u]['objcoeff'], 0, graph.degree[u], graph.nodes[v]['objcoeff'], 0, graph.degree[v], graph.edges[(u, v)]["coeff"]])

                            if u == v:
                                if (graph.nodes[v]['bias'] < 0.005):
                                    y.append(0)
                                else:
                                    y.append(1)

                                indices.append(num)
                        elif graph.nodes[u]['bipartite'] == 0 and graph.nodes[v]['bipartite'] == 1:
                            graph_new.add_node((u, v), type="VC", first = u, second = v, num=num)
                            features.append([graph.nodes[u]['objcoeff'], 0, graph.degree[u], 0, graph.nodes[v]['rhs'], graph.degree[v], graph.edges[(u, v)]["coeff"]])
                        elif graph.nodes[u]['bipartite'] == 1 and graph.nodes[v]['bipartite'] == 0:
                            graph_new.add_node((u, v), type="CV", first = u, second = v, num=num)
                            features.append([0, graph.nodes[u]['rhs'], graph.degree[u], graph.nodes[v]['objcoeff'], 0, graph.degree[v], graph.edges[(u, v)]["coeff"]])
                        elif graph.nodes[u]['bipartite'] == 1 and graph.nodes[v]['bipartite'] == 1:
                            graph_new.add_node((u, v), type="CC", first = u, second = v, num=num)
                            features.append([0, graph.nodes[u]['rhs'], graph.degree[u], 0, graph.nodes[v]['rhs'], graph.degree[v], graph.edges[(u, v)]["coeff"]])
                        num += 1

            for _, data in graph_new.nodes(data=True):
                first = data["first"]
                second = data["second"]
                num = data["num"]

                for n in graph.neighbors(first):
                    if graph.has_edge(n, second):
                        matrices_1.append([num, graph_new.nodes[(n, second)]["num"]])

                for n in graph.neighbors(second):
                    if graph.has_edge(first, n):
                        matrices_2.append([num, graph_new.nodes[(first, n)]["num"]])

            matrices_1 = torch.tensor(matrices_1).t().contiguous()
            matrices_2 = torch.tensor(matrices_2).t().contiguous()

            data = Data()
            data.node_features = torch.from_numpy(np.array(features)).to(torch.float)

            data.y = torch.from_numpy(np.array(y)).to(torch.long)

            data.edge_index_1 = matrices_1.to(torch.long)
            data.edge_index_2 = matrices_2.to(torch.long)

            data.indices = torch.from_numpy(np.array(indices)).to(torch.long)

            data.num = num

            data_list.append(data)



        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_1']:
            return self.num
        if key in ['edge_index_2']:
            return self.num
        if key in ['indices']:
            return self.num
        else:
            return 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


pathr = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'DS')
dataset = GraphDataset(pathr, 0.005, transform=MyTransform())  # .shuffle()
print("###")
print(dataset.data.y.sum() / dataset.data.y.size(-1))


exit()

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(hidden=64, aggr="mean").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=10,
                                                       min_lr=0.0000001)


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

for epoch in range(1, 50):
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
        break

    print('Epoch: {:03d}, LR: {:.7f}, Train Loss: {:.7f},  '
          'Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, lr, train_loss,
                                                                        train_acc, val_acc, test_acc))
