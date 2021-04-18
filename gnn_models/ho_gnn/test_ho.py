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

    def forward(self, source, target, edge_index, size):
        tmp = self.propagate(edge_index, x=source, size=size)

        out = self.mlp((1 + self.eps) * target + tmp)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

# class SimpleBipartiteLayer(MessagePassing):
#     def __init__(self, dim, aggr):
#         super(SimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")
#
#         self.nn = Sequential(Linear(2*dim, dim), ReLU(), Linear(dim, dim), ReLU(),
#                              BN(dim))
#
#     def forward(self, source, target, edge_index, size):
#         out = self.propagate(edge_index, x=source, t=target, size=size)
#
#         return out
#
#     def message(self, x_j, t_i):
#         return self.nn(torch.cat([t_i, x_j], dim=-1))
#
#     def __repr__(self):
#         return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden, aggr,):
        super(SimpleNet, self).__init__()

        # Embed initial node features.
        self.vv_node_encoder = Sequential(Linear(4, hidden), ReLU(), Linear(hidden, hidden))
        self.cc_node_encoder = Sequential(Linear(4, hidden), ReLU(), Linear(hidden, hidden))
        self.vc_node_encoder = Sequential(Linear(5, hidden), ReLU(), Linear(hidden, hidden))
        self.cv_node_encoder = Sequential(Linear(5, hidden), ReLU(), Linear(hidden, hidden))

        self.vv_cv_1_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cc_vc_1_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.vc_cc_1_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cv_vv_1_1 = SimpleBipartiteLayer(hidden, aggr=aggr)

        self.vv_vc_2_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cc_cv_2_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.vc_vv_2_1 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cv_cc_2_1 = SimpleBipartiteLayer(hidden, aggr=aggr)

        self.vv_joint_1 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.cc_joint_1 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.vc_joint_1 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.cv_joint_1 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        self.vv_cv_1_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cc_vc_1_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.vc_cc_1_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cv_vv_1_2 = SimpleBipartiteLayer(hidden, aggr=aggr)

        self.vv_vc_2_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cc_cv_2_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.vc_vv_2_2 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cv_cc_2_2 = SimpleBipartiteLayer(hidden, aggr=aggr)

        self.vv_joint_2 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.cc_joint_2 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.vc_joint_2 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.cv_joint_2 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        self.vv_cv_1_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cc_vc_1_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.vc_cc_1_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cv_vv_1_3 = SimpleBipartiteLayer(hidden, aggr=aggr)

        self.vv_vc_2_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cc_cv_2_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.vc_vv_2_3 = SimpleBipartiteLayer(hidden, aggr=aggr)
        self.cv_cc_2_3 = SimpleBipartiteLayer(hidden, aggr=aggr)

        self.vv_joint_3 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.cc_joint_3 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.vc_joint_3 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.cv_joint_3 = Sequential(Linear(2 * hidden, hidden), ReLU(), Linear(hidden, hidden))

        # MLP used for classification.
        self.lin1 = Linear(4 * hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.lin4 = Linear(hidden, 2)

    def forward(self, data):

        # Get data of batch.
        vv_node_features = data.vv_node_features
        cc_node_features = data.cc_node_features
        vc_node_features = data.vc_node_features
        cv_node_features = data.cv_node_features

        edge_index_vv_cv_1 = data.edge_index_vv_cv_1
        edge_index_vv_vc_2 = data.edge_index_vv_vc_2

        edge_index_cc_vc_1 = data.edge_index_cc_vc_1
        edge_index_cc_cv_2 = data.edge_index_cc_cv_2

        edge_index_vc_cc_1 = data.edge_index_vc_cc_1
        edge_index_vc_vv_2 = data.edge_index_vc_vv_2

        edge_index_cv_vv_1 = data.edge_index_cv_vv_1
        edge_index_cv_cc_2 = data.edge_index_cv_cc_2

        num_nodes_vv = data.num_nodes_vv
        num_nodes_cc = data.num_nodes_cc
        num_nodes_vc = data.num_nodes_vc
        num_nodes_cv = data.num_nodes_cv

        # Compute initial node embeddings.
        vv_0 = self.vv_node_encoder(vv_node_features)
        cc_0 = self.cc_node_encoder(cc_node_features)
        vc_0 = self.vc_node_encoder(vc_node_features)
        cv_0 = self.cv_node_encoder(cv_node_features)

        cv_1_1 = self.vv_cv_1_1(vv_0, cv_0, edge_index_vv_cv_1, [num_nodes_vv.sum(), num_nodes_cv.sum()])
        cc_1_1 = self.vc_cc_1_1(vc_0, cc_0, edge_index_vc_cc_1, [num_nodes_vc.sum(), num_nodes_cc.sum()])
        vc_1_1 = self.cc_vc_1_1(cc_0, vc_0, edge_index_cc_vc_1, [num_nodes_cc.sum(), num_nodes_vc.sum()])
        vv_1_1 = self.cv_vv_1_1(cv_0, vv_0, edge_index_cv_vv_1, [num_nodes_cv.sum(), num_nodes_vv.sum()])

        vc_2_1 = self.vv_vc_2_1(vv_0, vc_0, edge_index_vv_vc_2, [num_nodes_vv.sum(), num_nodes_vc.sum()])
        vv_2_1 = self.vc_vv_2_1(vc_0, vv_0, edge_index_vc_vv_2, [num_nodes_vc.sum(), num_nodes_vv.sum()])
        cv_2_1 = self.cc_cv_2_1(cc_0, cv_0, edge_index_cc_cv_2, [num_nodes_cc.sum(), num_nodes_cv.sum()])
        cc_2_1 = self.cv_cc_2_1(cv_0, cc_0, edge_index_cv_cc_2, [num_nodes_cv.sum(), num_nodes_cc.sum()])

        vv_1 = self.vv_joint_1(torch.cat([vv_1_1, vv_2_1], dim=-1))
        cc_1 = self.vv_joint_1(torch.cat([cc_1_1, cc_2_1], dim=-1))
        vc_1 = self.vv_joint_1(torch.cat([vc_1_1, vc_2_1], dim=-1))
        cv_1 = self.vv_joint_1(torch.cat([cv_1_1, cv_2_1], dim=-1))


        cv_1_2 = self.vv_cv_1_2(vv_1, cv_1, edge_index_vv_cv_1, [num_nodes_vv.sum(), num_nodes_cv.sum()])
        cc_1_2 = self.vc_cc_1_2(vc_1, cc_1, edge_index_vc_cc_1, [num_nodes_vc.sum(), num_nodes_cc.sum()])
        vc_1_2 = self.cc_vc_1_2(cc_1, vc_1, edge_index_cc_vc_1, [num_nodes_cc.sum(), num_nodes_vc.sum()])
        vv_1_2 = self.cv_vv_1_2(cv_1, vv_1, edge_index_cv_vv_1, [num_nodes_cv.sum(), num_nodes_vv.sum()])

        vc_2_2 = self.vv_vc_2_2(vv_1, vc_1, edge_index_vv_vc_2, [num_nodes_vv.sum(), num_nodes_vc.sum()])
        vv_2_2 = self.vc_vv_2_2(vc_1, vv_1, edge_index_vc_vv_2, [num_nodes_vc.sum(), num_nodes_vv.sum()])
        cv_2_2 = self.cc_cv_2_2(cc_1, cv_1, edge_index_cc_cv_2, [num_nodes_cc.sum(), num_nodes_cv.sum()])
        cc_2_2 = self.cv_cc_2_2(cv_1, cc_1, edge_index_cv_cc_2, [num_nodes_cv.sum(), num_nodes_cc.sum()])

        vv_2 = self.vv_joint_2(torch.cat([vv_1_2, vv_2_2], dim=-1))
        cc_2 = self.vv_joint_2(torch.cat([cc_1_2, cc_2_2], dim=-1))
        vc_2 = self.vv_joint_2(torch.cat([vc_1_2, vc_2_2], dim=-1))
        cv_2 = self.vv_joint_2(torch.cat([cv_1_2, cv_2_2], dim=-1))


        cv_1_3 = self.vv_cv_1_3(vv_2, cv_2, edge_index_vv_cv_1, [num_nodes_vv.sum(), num_nodes_cv.sum()])
        cc_1_3 = self.vc_cc_1_3(vc_2, cc_2, edge_index_vc_cc_1, [num_nodes_vc.sum(), num_nodes_cc.sum()])
        vc_1_3 = self.cc_vc_1_3(cc_2, vc_2, edge_index_cc_vc_1, [num_nodes_cc.sum(), num_nodes_vc.sum()])
        vv_1_3 = self.cv_vv_1_3(cv_2, vv_2, edge_index_cv_vv_1, [num_nodes_cv.sum(), num_nodes_vv.sum()])

        vc_2_3 = self.vv_vc_2_3(vv_2, vc_2, edge_index_vv_vc_2, [num_nodes_vv.sum(), num_nodes_vc.sum()])
        vv_2_3 = self.vc_vv_2_3(vc_2, vv_2, edge_index_vc_vv_2, [num_nodes_vc.sum(), num_nodes_vv.sum()])
        cv_2_3 = self.cc_cv_2_3(cc_2, cv_2, edge_index_cc_cv_2, [num_nodes_cc.sum(), num_nodes_cv.sum()])
        cc_2_3 = self.cv_cc_2_3(cv_2, cc_2, edge_index_cv_cc_2, [num_nodes_cv.sum(), num_nodes_cc.sum()])

        vv_3 = self.vv_joint_1(torch.cat([vv_1_3, vv_2_3], dim=-1))
        cc_3 = self.vv_joint_1(torch.cat([cc_1_3, cc_2_3], dim=-1))
        vc_3 = self.vv_joint_1(torch.cat([vc_1_3, vc_2_3], dim=-1))
        cv_3 = self.vv_joint_1(torch.cat([cv_1_3, cv_2_3], dim=-1))

        x = torch.cat([vv_0, vv_1, vv_2, vv_3], dim=-1)

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
        num_graphs = len(os.listdir(data_path))

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(data_path)):
            print(num)
            # Get graph.
            graph = nx.read_gpickle(data_path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_undirected() if nx.is_directed(graph) else graph

            graph_new = nx.Graph()

            matrices_vv_cv_1 = []
            matrices_vv_vc_2 = []

            matrices_cc_vc_1 = []
            matrices_cc_cv_2 = []

            matrices_vc_cc_1 = []
            matrices_vc_vv_2 = []

            matrices_cv_vv_1 = []
            matrices_cv_cc_2 = []

            features_vv = []
            features_cc = []
            features_vc = []
            features_cv = []

            num_vv = 0
            num_cc = 0
            num_vc = 0
            num_cv = 0

            y = []

            c = 0
            for i, (u, v) in enumerate(graph.edges):
                if graph.nodes[u]['bipartite'] == 0:
                    graph_new.add_node((u, v), type="VC", first=u, second=v, num=num_vc, feauture = [graph.nodes[u]['objcoeff'], graph.degree[u], graph.nodes[v]['rhs'],  graph.degree[v], graph.edges[(u,v)]["coeff"]])
                    features_vc.append([graph.nodes[u]['objcoeff'], graph.degree[u], graph.nodes[v]['rhs'],  graph.degree[v], graph.edges[(u,v)]["coeff"]])
                    c += 1
                    graph_new.add_node((v, u), type="CV", first=v, second=u, num=num_cv, feauture = [graph.nodes[v]['rhs'],  graph.degree[v], graph.nodes[u]['objcoeff'], graph.degree[u], graph.edges[(u,v)]["coeff"]])
                    c += 1
                    features_cv.append([graph.nodes[v]['rhs'],  graph.degree[v], graph.nodes[u]['objcoeff'], graph.degree[u], graph.edges[(u,v)]["coeff"]])

                    num_vc += 1
                    num_cv += 1

            for i, v in enumerate(graph.nodes):
                if graph.nodes[v]['bipartite'] == 0:
                    graph_new.add_node((v, v), type="VV", first=v, second=v, num=num_vv,
                                       feauture=[graph.nodes[v]['objcoeff'], graph.degree[v],
                                                 graph.nodes[v]['objcoeff'], graph.degree[v]])
                    features_vv.append(
                        [graph.nodes[v]['objcoeff'], graph.degree[v], graph.nodes[v]['objcoeff'], graph.degree[v]])
                    c += 1
                    num_vv += 1

                    if (graph.nodes[v]['bias'] < 0.005):
                        y.append(0)
                    else:
                        y.append(1)
                elif graph.nodes[v]['bipartite'] == 1:
                    graph_new.add_node((v, v), type="CC", first=v, second=v, num=num_cc,
                                       feauture=[graph.nodes[v]['rhs'], graph.degree[v], graph.nodes[v]['rhs'],
                                                 graph.degree[v]])
                    features_cc.append([graph.nodes[v]['rhs'], graph.degree[v], graph.nodes[v]['rhs'], graph.degree[v]])
                    c += 1
                    num_cc += 1

            for i, (v, data) in enumerate(graph_new.nodes(data=True)):
                first = data["first"]
                second = data["second"]
                num = data["num"]

                for n in graph.neighbors(first):
                    if graph_new.nodes[v]["type"] == "VV":
                        if graph.has_edge(n, second):
                            matrices_vv_cv_1.append([num, graph_new.nodes[(n, second)]["num"]])
                    if graph_new.nodes[v]["type"] == "CC":
                        if graph.has_edge(n, second):
                            matrices_cc_vc_1.append([num, graph_new.nodes[(n, second)]["num"]])
                    if graph_new.nodes[v]["type"] == "VC":
                        if graph.has_edge(n, second):
                            matrices_vc_cc_1.append([num, graph_new.nodes[(n, second)]["num"]])
                    if graph_new.nodes[v]["type"] == "CV":
                        if graph.has_edge(n, second):
                            matrices_cv_vv_1.append([num, graph_new.nodes[(n, second)]["num"]])

                for n in graph.neighbors(second):
                    if graph_new.nodes[v]["type"] == "VV":
                        if graph.has_edge(first, n):
                            matrices_vv_vc_2.append([num, graph_new.nodes[(first, n)]["num"]])
                    if graph_new.nodes[v]["type"] == "CC":
                        if graph.has_edge(first, n):
                            matrices_cc_cv_2.append([num, graph_new.nodes[(first, n)]["num"]])
                    if graph_new.nodes[v]["type"] == "VC":
                        if graph.has_edge(first, n):
                            matrices_vc_vv_2.append([num, graph_new.nodes[(first, n)]["num"]])
                    if graph_new.nodes[v]["type"] == "CV":
                        if graph.has_edge(first, n):
                            matrices_cv_cc_2.append([num, graph_new.nodes[(first, n)]["num"]])

            matrices_vv_cv_1 = torch.tensor(matrices_vv_cv_1).t().contiguous()
            matrices_vv_vc_2 = torch.tensor(matrices_vv_vc_2).t().contiguous()

            matrices_cc_vc_1 = torch.tensor(matrices_cc_vc_1).t().contiguous()
            matrices_cc_cv_2 = torch.tensor(matrices_cc_cv_2).t().contiguous()

            matrices_vc_cc_1 = torch.tensor(matrices_vc_cc_1).t().contiguous()
            matrices_vc_vv_2 = torch.tensor(matrices_vc_vv_2).t().contiguous()

            matrices_cv_vv_1 = torch.tensor(matrices_cv_vv_1).t().contiguous()
            matrices_cv_cc_2 = torch.tensor(matrices_cv_cc_2).t().contiguous()

            data = Data()

            data.vv_node_features = torch.from_numpy(np.array(features_vv)).to(torch.float)
            data.cc_node_features = torch.from_numpy(np.array(features_cc)).to(torch.float)
            data.vc_node_features = torch.from_numpy(np.array(features_vc)).to(torch.float)
            data.cv_node_features = torch.from_numpy(np.array(features_cv)).to(torch.float)

            data.y = torch.from_numpy(np.array(y)).to(torch.long)

            data.edge_index_vv_cv_1 = matrices_vv_cv_1.to(torch.long)
            data.edge_index_vv_vc_2 = matrices_vv_vc_2.to(torch.long)

            data.edge_index_cc_vc_1 = matrices_cc_vc_1.to(torch.long)
            data.edge_index_cc_cv_2 = matrices_cc_cv_2.to(torch.long)

            data.edge_index_vc_cc_1 = matrices_vc_cc_1.to(torch.long)
            data.edge_index_vc_vv_2 = matrices_vc_vv_2.to(torch.long)

            data.edge_index_cv_vv_1 = matrices_cv_vv_1.to(torch.long)
            data.edge_index_cv_cc_2 = matrices_cv_cc_2.to(torch.long)

            data.num_nodes_vv = num_vv
            data.num_nodes_cc = num_cc
            data.num_nodes_vc = num_vc
            data.num_nodes_cv = num_cv

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Preprocess indices of bipartite graphs to make batching work.
class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_vv_cv_1']:
            return torch.tensor([self.num_nodes_vv, self.num_nodes_cv]).view(2, 1)
        if key in ['edge_index_vv_cv_2']:
            return torch.tensor([self.num_nodes_vv, self.num_nodes_cv]).view(2, 1)
        if key in ['edge_index_cc_vc_1']:
            return torch.tensor([self.num_nodes_cc, self.num_nodes_vc]).view(2, 1)
        if key in ['edge_index_cc_cv_2']:
            return torch.tensor([self.num_nodes_cc, self.num_nodes_cv]).view(2, 1)
        if key in ['edge_index_vc_cc_1']:
            return torch.tensor([self.num_nodes_vc, self.num_nodes_cc]).view(2, 1)
        if key in ['edge_index_vc_vv_2']:
            return torch.tensor([self.num_nodes_vc, self.num_nodes_vv]).view(2, 1)
        if key in ['edge_index_cv_vv_1']:
            return torch.tensor([self.num_nodes_cv, self.num_nodes_vv]).view(2, 1)
        if key in ['edge_index_cv_cc_2']:
            return torch.tensor([self.num_nodes_cv, self.num_nodes_cc]).view(2, 1)
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
