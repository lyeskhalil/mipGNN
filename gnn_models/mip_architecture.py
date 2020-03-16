import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch_geometric

class GISDS(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISDS, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ERS"

    @property
    def processed_file_names(self):
        return "ERS"

    def download(self):
        pass

    def process(self):
        data_list = []

        path = '../gisp_generator/DATA/er_200/'

        total = len(os.listdir(path))

        for num, filename in enumerate(os.listdir(path)):
            print(filename)

            graph = nx.read_gpickle(path + filename)

            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            data = Data(edge_index=edge_index)

            y = []
            node_type = []

            assoc_var = []
            assoc_con = []

            var_feat = []
            con_feat = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):

                # Node is a variable.
                if node_data['bipartite'] == 0:
                    y.append(node_data['bias'])
                    node_type.append(0)
                    assoc_var.append(i)
                    coeff = node_data['objcoeff']

                    # TODO: Maybe scale this
                    var_feat.append([coeff/100.0, graph.degree[i]])
                # Node is constraint.
                else:
                    node_type.append(1)
                    assoc_con.append(i)
                    rhs = node_data['rhs']
                    con_feat.append([rhs, graph.degree[i]])

            y = torch.from_numpy(np.array(y)).to(torch.float).to(torch.float)
            data.y = y
            data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
            data.node_types = torch.from_numpy(np.array(node_type)).to(torch.long)
            data.assoc_var = torch.from_numpy(np.array(assoc_var)).to(torch.long)
            data.assoc_con = torch.from_numpy(np.array(assoc_con)).to(torch.long)


            edge_types = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):

                if graph.nodes[s]['bipartite']:
                    edge_types.append([0, edge_data['coeff']])
                else:
                    edge_types.append([1, edge_data['coeff']])

            data.edge_types = torch.from_numpy(np.array(edge_types)).to(torch.float)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        return self.num_nodes if key in [
            'edge_index', 'assoc_var', 'assoc_con'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        new_data.num_nodes = data.node_types.size(0)
        return new_data
