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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


import torch_geometric.utils.softmax
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

class SimpleBipartiteLayer(MessagePassing):
    def __init__(self, dim, aggr):
        super(SimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        self.nn = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                             BN(dim))

    def forward(self, source, target, edge_index, size):
        out = self.propagate(edge_index, x=source, t=target, size=size)

        return out

    def message(self, x_j, t_i):
        return self.nn(torch.cat([t_i, x_j], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden, aggr, num_layers):
        super(SimpleNet, self).__init__()
        self.num_layers = num_layers

        # Embed initial node features.
        self.vv_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.cc_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.vc_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.cv_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))

        self.vv_cv_1 = SimpleBipartiteLayer(hidden, aggr=aggr)

        # MLP used for classification.
        # self.lin1 = Linear((num_layers + 1) * hidden, hidden)
        # self.lin2 = Linear(hidden, hidden)
        # self.lin3 = Linear(hidden, hidden)
        # self.lin4 = Linear(hidden, 2)

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

        # Compute initial node embeddings.



        var_node_features_0 = self.vv_node_encoder(vv_node_features)





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
        for num, filename in enumerate(os.listdir(data_path)[0:2]):
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
                    graph_new.add_node((u, v), type="VC", first=u, second=v, num=c, feauture = [graph.nodes[u]['objcoeff'], graph.degree[u], graph.nodes[v]['rhs'],  graph.degree[v], graph.edges[(u,v)]["coeff"]])
                    features_vc.append([graph.nodes[u]['objcoeff'], graph.degree[u], graph.nodes[v]['rhs'],  graph.degree[v], graph.edges[(u,v)]["coeff"]])
                    c += 1
                    graph_new.add_node((v, u), type="CV", first=v, second=u, num=c, feauture = [graph.nodes[v]['rhs'],  graph.degree[v], graph.nodes[u]['objcoeff'], graph.degree[u], graph.edges[(u,v)]["coeff"]])
                    c += 1
                    features_cv.append([graph.nodes[v]['rhs'],  graph.degree[v], graph.nodes[u]['objcoeff'], graph.degree[u], graph.edges[(u,v)]["coeff"]])

                    num_vc += 1
                    num_cv += 1
                for i, v in enumerate(graph.nodes):
                 if graph.nodes[v]['bipartite'] == 0:
                     graph_new.add_node((v,v), type="VV", first=v, second=v, num=c, feauture = [graph.nodes[v]['objcoeff'], graph.degree[v], graph.nodes[v]['objcoeff'], graph.degree[v]])
                     features_vv.append([graph.nodes[v]['objcoeff'], graph.degree[v], graph.nodes[v]['objcoeff'], graph.degree[v]])
                     c += 1
                     num_vv += 1

                     if (graph.nodes[v]['bias'] < 0.005):
                         y.append(0)
                     else:
                         y.append(1)
                 elif graph.nodes[v]['bipartite'] == 1:
                     graph_new.add_node((v,v), type="CC", first=v, second=v, num=c,  feauture = [graph.nodes[v]['rhs'], graph.degree[v], graph.nodes[v]['rhs'], graph.degree[v]])
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
                            # Source node is var. VV->CV
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

            data.edge_index_vv_cv_1 = matrices_vv_cv_1
            data.edge_index_vv_vc_2 = matrices_vv_vc_2

            data.edge_index_cc_vc_1 = matrices_cc_vc_1
            data.edge_index_cc_cv_2 = matrices_cc_cv_2

            data.edge_index_vc_cc_1 = matrices_vc_cc_1
            data.edge_index_vc_vv_2 = matrices_vc_vv_2

            data.edge_index_cv_vv_1 = matrices_cv_vv_1
            data.edge_index_cv_cc_2 = matrices_cv_cc_2

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
print(dataset.data.vv_node_features.size())