import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
from pathlib import Path

import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
#from gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet

# TODO: Uncomment for second model.
#from gnn_models.EdgeConv.mip_bipartite_class_test import SimpleNet


def get_prediction(model_name, graph, bias_threshold=0.00):

    if model_name[0:3] == "ECS":
        from gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet
    if model_name[0:3] == "EC_":
        from gnn_models.EdgeConv.mip_bipartite_class import SimpleNet
    if model_name[0:4] == "GINS":
        from gnn_models.GIN.mip_bipartite_simple_class import SimpleNet
    if model_name[0:4] == "GIN_":
        from gnn_models.GIN.mip_bipartite_class import SimpleNet
    if model_name[0:3] == "SGS":
        from gnn_models.Sage.mip_bipartite_simple_class import SimpleNet
    if model_name[0:3] == "SG_":
        from gnn_models.Sage.mip_bipartite_class import SimpleNet



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNet(64, aggr="mean", num_layers=4).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))

    # data, var_node, node_var = create_data_object(graph)
    data, node_to_varnode, _ = create_data_object(graph, 0.0)
    model.eval()

    data = data.to(device)

    # out = model(data).max(dim=1)[1].cpu().detach().numpy()
    out = model(data).exp()[:,1].cpu().detach().numpy()

    #y_real = data.y_real.cpu().detach().numpy()
    # return out, node_var, var_node
    return out, node_to_varnode#, y_real

def get_variable_cpxid(graph, node_to_varnode, prediction):
    node_names_list = list(graph.nodes())
    dict_varname_seqid = {}
    for var_graphid, var_seqid in node_to_varnode.items():
        # print(var_graphid, var_seqid)
        dict_varname_seqid[node_names_list[var_graphid]] = (var_seqid, prediction[var_seqid])

    return dict_varname_seqid

def create_data_object(graph, bias_threshold):
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

            if 'bias' in node_data and node_data['bias'] is not None:
                y_real.append(node_data['bias'])
                if (node_data['bias'] <= bias_threshold):
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

    return data, node_to_varnode, node_to_connode

# pd = "../data_new/data_graphsonly/gisp/p_hat300-2.clq/train/"
#pd = "../datagen/data/gisp/p_hat300-2.clq/test/"
#filename = os.listdir(pd)[0]
#graph = nx.read_gpickle(pd + filename)
#
#
#out, node_to_varnode, y_real =get_prediction("../gnn_models/EdgeConv/trained_p_hat300-2", graph)
#
#print(out, out.mean())
#print(y_real)
#

#dict_results = get_variable_cpxid(graph, node_to_varnode, out)
#print(dict_results)

#y_prediction = [out[var_seqid] for 

#loss = torch.nn.MSELoss()
#output = loss(torch.from_numpy(out), torch.from_numpy(y_real))
#print(output)
#loss = torch.nn.L1Loss()
#output = loss(torch.from_numpy(out), torch.from_numpy(y_real))
#print(output)
#np.random.shuffle(out)
#output = loss(torch.from_numpy(out), torch.from_numpy(y_real))
#print(output)
