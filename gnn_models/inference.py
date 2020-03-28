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
from gnn_models.mip_alternating_arch import Net


def get_prediction(model_name, graph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(dim=64).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))

    data = create_data_object(graph)

    model.eval()

    data = data.to(device)
    out = model(data).cpu().detach().numpy()

    return out

def create_data_object(graph):
    # Make graph directed.
    graph = nx.convert_node_labels_to_integers(graph)
    graph = graph.to_directed() if not nx.is_directed(graph) else graph
    data = Data()

    #  Map for new nodes in graph.

    # Original node to var nodes
    var_node = {}
    # Var node to orignal
    node_var = {}

    con_node = {}



    # Number of variables.
    var_i = 0
    # Number of constraints.
    con_i = 0
    # Targets
    y = []
    # Features for variable nodes.
    var_feat = []
    # Feature for constraints nodes.
    con_feat = []
    # Right-hand sides of equations.
    rhss = []
    # Sums over coefficients.
    a_sum = []
    for i, (node, node_data) in enumerate(graph.nodes(data=True)):
        # Node is a variable.
        if node_data['bipartite'] == 0:
            var_node[i] = var_i
            node_var[var_i] = i
            var_i += 1

            y.append(node_data['bias'])
            # TODO: Scaling meaingful?
            var_feat.append([node_data['objcoeff'] / 100.0, graph.degree[i]])

        # Node is constraint.
        else:
            a = []
            for e in graph.edges(node, data=True):
                a.append(graph[e[0]][e[1]]['coeff'])
            a_sum.append(sum(a))

            con_node[i] = con_i
            con_i += 1

            rhs = node_data['rhs']
            rhss.append(rhs)
            con_feat.append([rhs, graph.degree[i]])

    num_nodes_var = var_i
    num_nodes_con = con_i
    # Edge list for var->con graphs.
    edge_list_var = []
    # Edge list for con->var graphs.
    edge_list_con = []

    edge_features_var = []
    edge_features_con = []
    for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
        # Source node is con, target node is var.
        if graph.nodes[s]['bipartite'] == 1:
            edge_list_con.append([con_node[s], var_node[t]])
            edge_features_con.append([edge_data['coeff']])
        else:
            edge_list_var.append([var_node[s], con_node[t]])
            edge_features_var.append([edge_data['coeff']])

    edge_index_var = torch.tensor(edge_list_var).t().contiguous()
    edge_index_con = torch.tensor(edge_list_con).t().contiguous()

    data.edge_index_var = edge_index_var
    data.edge_index_con = edge_index_con
    data.y = torch.from_numpy(np.array(y)).to(torch.float)
    data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
    data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
    data.rhs = torch.from_numpy(np.array(rhss)).to(torch.float)
    data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
    data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)
    data.asums = torch.from_numpy(np.array(a_sum)).to(torch.float)
    data.num_nodes_var = num_nodes_var
    data.num_nodes_con = num_nodes_con

    return data, var_node, node_var