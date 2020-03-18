import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx


import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

torch.autograd.set_detect_anomaly(True)

from gnn_models.mip_alternating_arch import Net

class GISR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ERBIN"

    @property
    def processed_file_names(self):
        return "ERBIN"

    def download(self):
        pass

    def process(self):
        data_list = []

        path = '../gisp_generator/DATA/er_200/'

        total = len(os.listdir(path))

        for num, filename in enumerate(os.listdir(path)):
            print(filename, num, total)

            graph = nx.read_gpickle(path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            data = Data()

            # Compute map for new nodes in graph.
            var_node = {}
            con_node = {}
            var_i = 0
            con_i = 0
            y = []
            var_feat = []
            con_feat = []
            rhss = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable.
                if node_data['bipartite'] == 0:
                    var_node[i] = var_i
                    var_i += 1

                    y.append(node_data['bias'])
                    coeff = node_data['objcoeff']

                    # TODO: Scaling meaingful?
                    var_feat.append([coeff / 100.0, graph.degree[i]])

                # Node is constraint.
                else:
                    con_node[i] = con_i
                    con_i += 1

                    rhs = node_data['rhs']
                    rhss.append(rhs)
                    con_feat.append([rhs, graph.degree[i]])

            num_nodes_var = var_i
            num_nodes_con = con_i
            edge_list_var = []
            edge_list_con = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
                # Source node is con, target node is var.
                if graph.nodes[s]['bipartite'] == 1:
                    edge_list_var.append([con_node[s],var_node[t]])
                else:
                    edge_list_con.append([var_node[s],con_node[t]])

            edge_index_var = torch.tensor(list(edge_list_var)).t().contiguous()
            edge_index_con = torch.tensor(list(edge_list_con)).t().contiguous()

            edge_features_var = []
            edge_features_con = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
                # Source node is con, target node is var.
                if graph.nodes[s]['bipartite'] == 1:
                    edge_features_var.append([edge_data['coeff']])
                else:
                    edge_features_con.append([edge_data['coeff']])


            y = torch.from_numpy(np.array(y)).to(torch.float).to(torch.float)
            data.y = y
            data.edge_index_var = edge_index_var
            data.edge_index_con = edge_index_con
            data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
            data.rhs = torch.from_numpy(np.array(rhss)).to(torch.float)

            data.num_nodes_var = torch.LongTensor(num_nodes_var)
            data.num_nodes_con = torch.LongTensor(num_nodes_con)

            data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
            data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_var']:
            return torch.tensor([self.num_nodes_var, self.num_nodes_con])
        elif key in ['edge_index_con']:
            return  torch.tensor([self.num_nodes_con, self.num_nodes_var])
        else:
            return 0

class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        return new_data

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DS')
dataset = GISR(path, transform=MyTransform()).shuffle()
dataset.data.y = torch.log(dataset.data.y + 1.0)
print(len(dataset))




