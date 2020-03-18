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

from gnn_models.mip_architecture import Net

class GISR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ER"

    @property
    def processed_file_names(self):
        return "ERB"

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

            assoc_var = []
            assoc_con = []

            var_feat = []
            con_feat = []
            rhss = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable.
                if node_data['bipartite'] == 0:
                    var_node[i] = var_i
                    var_i += 1

                    y.append(node_data['bias'])
                    assoc_var.append(i)
                    coeff = node_data['objcoeff']

                    # TODO: Scaling meaingful?
                    var_feat.append([coeff / 100.0, graph.degree[i]])

                # Node is constraint.
                else:
                    con_node[i] = con_i
                    con_i += 1

                    assoc_con.append(i)
                    rhs = node_data['rhs']
                    rhss.append(rhs)
                    con_feat.append([rhs, graph.degree[i]])

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
            data.assoc_var = torch.from_numpy(np.array(assoc_var)).to(torch.long)
            data.assoc_con = torch.from_numpy(np.array(assoc_con)).to(torch.long)
            data.rhs = torch.from_numpy(np.array(rhss)).to(torch.float)

            data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
            data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)

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


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DS')
dataset = GISR(path, transform=MyTransform()).shuffle()
dataset.data.y = torch.log(dataset.data.y + 1.0)
print(len(dataset))
