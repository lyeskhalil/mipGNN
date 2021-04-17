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
        return "test"

    @property
    def processed_file_names(self):
        return "test"

    def download(self):
        pass

    def process(self):
        print("Preprocessing.")

        data_list = []
        num_graphs = len(os.listdir(data_path))

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(data_path)[0:5]):
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

            c = 0
            for i, (u, v) in enumerate(graph.edges):
                if graph.nodes[u]['bipartite'] == 0:
                    graph_new.add_node((u, v), type="VC", first=u, second=v, num=c)
                    c += 1
                    graph_new.add_node((v, u), type="CV", first=v, second=u, num=c)
                    c += 1

            for i, v in enumerate(graph.nodes):
                 if graph.nodes[v]['bipartite'] == 0:
                     graph_new.add_node((v,v), type="VV", first=v, second=v, num=c)
                     c += 1
                 elif graph.nodes[v]['bipartite'] == 1:
                     graph_new.add_node((v,v), type="CC", first=v, second=v, num=c)
                     c += 1

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
            print(matrices_vv_cv_1.size())
            matrices_vv_vc_2 = torch.tensor(matrices_vv_vc_2).t().contiguous()

            matrices_cc_vc_1 = torch.tensor(matrices_cc_vc_1).t().contiguous()
            matrices_cc_cv_2 = torch.tensor(matrices_cc_cv_2).t().contiguous()

            matrices_vc_cc_1 = torch.tensor(matrices_vc_cc_1).t().contiguous()
            matrices_vc_vv_2 = torch.tensor(matrices_vc_vv_2).t().contiguous()

            matrices_cv_vv_1 = torch.tensor(matrices_cv_vv_1).t().contiguous()
            matrices_cv_cc_2 = torch.tensor(matrices_cv_cc_2).t().contiguous()

            data = Data()

            data.matrices_vv_cv_1 = matrices_vv_cv_1
            # data.matrices_vv_vc_2 = matrices_vv_vc_2
            #
            # data.matrices_cc_vc_1 = matrices_cc_vc_1
            # data.matrices_cc_cv_2 = matrices_cc_cv_2
            #
            # data.matrices_vc_cc_1 = matrices_vc_cc_1
            # data.matrices_vc_vv_2 = matrices_vc_vv_2
            #
            # data.matrices_cv_vv_1 = matrices_cv_vv_1
            # data.matrices_cv_cc_2 = matrices_cv_cc_2

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = GraphDataset(".", 0.005, transform=None)  # .shuffle()