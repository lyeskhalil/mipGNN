import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
import argparse

import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
from gnn_models.mip_bipartite_arch import SimpleNet

import cplex

import callbacks_cplex

def get_prediction(model_name, graph, bias_threshold=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNet(128).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))

    # data, var_node, node_var = create_data_object(graph)
    data, node_to_varnode, _ = create_data_object(graph, bias_threshold)
    model.eval()

    data = data.to(device)

    # out = model(data).max(dim=1)[1].cpu().detach().numpy()
    out = model(data).exp()[:,1].cpu().detach().numpy()

    # return out, node_var, var_node
    return out, node_to_varnode

def get_variable_cpxid(graph, node_to_varnode, prediction):
    node_names_list = list(graph.nodes())
    dict_varname_seqid = {}
    dict_graphid_varname = {}
    for var_graphid, var_seqid in node_to_varnode.items():
        dict_varname_seqid[node_names_list[var_graphid]] = (var_seqid, prediction[var_seqid])
        dict_graphid_varname[var_seqid] = node_names_list[var_graphid]

    return dict_varname_seqid, dict_graphid_varname

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

            if (node_data['bias'] < bias_threshold):
                y.append(0)
            else:
                y.append(1)

            feat_var.append([node_data['objcoeff'], graph.degree[i]])
            obj.append([node_data['objcoeff']])
            index_var.append(0)

        # Node is constraint node.
        elif node_data['bipartite'] == 1:
            node_to_connode[i] = num_nodes_con
            num_nodes_con += 1

            rhs = node_data['rhs']
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

def create_data_object_old(graph):
    # Make graph directed.
    graph = nx.convert_node_labels_to_integers(graph)
    graph = graph.to_directed() if not nx.is_directed(graph) else graph
    data = Data()
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    data = Data(edge_index=edge_index)

    #  Map for new nodes in graph.

    # Original node to var nodes
    var_node = {}
    # Var node to orignal
    node_var = {}

    con_node = {}

    assoc_var = []
    assoc_con = []
    node_type = []
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
            node_type.append(0)
            assoc_var.append(i)
            node_var[var_i] = i
            var_i += 1

            if (node_data['bias'] < 0.05):
                y.append(0)
            else:
                y.append(1)
            # TODO: Scaling meaingful?
            var_feat.append([node_data['objcoeff'] / 100.0, graph.degree[i]])

        # Node is constraint.
        else:

            node_type.append(1)
            assoc_con.append(i)
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

    data.y = torch.from_numpy(np.array(y)).to(torch.long)
    #data.y = torch.from_numpy(np.array(y)).to(torch.float)

    data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
    data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
    #data.rhs = torch.from_numpy(np.array(rhss)).to(torch.float)
    #data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
    #data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)
    data.asums = torch.from_numpy(np.array(a_sum)).to(torch.float)
    data.num_nodes_var = num_nodes_var
    data.num_nodes_con = num_nodes_con
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

    return data, var_node, node_var

def set_cplex_priorities(instance_cpx, prediction, dict_varname_seqid, dict_graphid_varname):
    # score variables based on bias prediction
    scores = np.max((-(1-prediction), -prediction), axis=0)
    priorities = np.argsort(scores)

    # set priorities
    # reference: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.OrderInterface-class.html
    order_tuples = []
    # node_names_list = list(graph.nodes())
    # for priority, var_idx_model in enumerate(priorities):
    #     var_idx = node_var[var_idx_model]
    #     var_name = node_names_list[var_idx]
    #     order_tuples += [(var_name, priority, instance_cpx.order.branch_direction.up)]

    # for var_name in dict_varname_seqid:
    #     var_graphid, var_prediction = dict_varname_seqid[var_name]
    #     order_tuples += [(var_name, priority, instance_cpx.order.branch_direction.up)]

    for priority, var_graphid in enumerate(priorities):
        var_name = dict_graphid_varname[var_graphid]
        order_tuples += [(var_name, priority, instance_cpx.order.branch_direction.up)]


    instance_cpx.order.set(order_tuples)
    # print(instance_cpx.order.get())

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default='default')
    parser.add_argument("-instance", type=str, default='er_200_SET2_1k/er_n=200_m=1867_p=0.10_SET2_setparam=100.00_alpha=0.75_606')
    parser.add_argument("-model", type=str, default='../gnn_models/trained_model_er_200_SET2_1k_new')
    parser.add_argument("-barebones", type=bool, default=True)


    # Parameters for exact local branching
    parser.add_argument("-elb_threshold", type=int, default=5)

    args = parser.parse_args()

    instance_name = args.instance

    """ Create CPLEX instance """
    instance_cpx = cplex.Cplex("../gisp_generator/LP/" + instance_name + ".lp")

    """ Set CPLEX parameters, if any """
    instance_cpx.parameters.emphasis.mip.set(1)
    if args.barebones:
        instance_cpx.parameters.mip.limits.cutpasses.set(-1)
        instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
        instance_cpx.parameters.preprocessing.presolve.set(0)

    """ CPLEX output management """
    instance_cpx.set_log_stream(sys.stdout)
    instance_cpx.set_results_stream(sys.stdout)

    """ Solve CPLEX instance with user-selected method """
    if args.method == 'default':
        instance_cpx.solve()

    else:
        """ Read in the pickled graph and the trained model """
        graph = nx.read_gpickle("../gisp_generator/DATA/" + instance_name + ".pk")
        prediction, node_to_varnode = get_prediction(model_name=args.model, graph=graph)
        dict_varname_seqid, dict_graphid_varname = get_variable_cpxid(graph, node_to_varnode, prediction)
        print(prediction)
        # todo check dimensions of p

        num_variables = instance_cpx.variables.get_num()

        if 'local_branching' in args.method:
            scores = np.max(((1-prediction), prediction), axis=0)
            local_branching_coeffs = [list(range(len(prediction))), scores.tolist()]

        if args.method == 'default_emptycb':
            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_empty)

            instance_cpx.solve()

        elif args.method == 'branching_priorities':
            set_cplex_priorities(instance_cpx, prediction, dict_varname_seqid, dict_graphid_varname)
            instance_cpx.solve()

        elif args.method == 'local_branching_approx':
            instance_cpx.linear_constraints.add(
                lin_expr=[local_branching_coeffs],
                senses=['L'],
                rhs=[args.elb_threshold],
                names=['local_branching'])

            instance_cpx.solve()
        
        elif args.method == 'local_branching_exact':
            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_local_exact)

            branch_cb.coeffs = local_branching_coeffs
            branch_cb.threshold = args.elb_threshold
            branch_cb.is_root = True

            instance_cpx.solve()

        elif args.method == 'node_selection':
            # score variables based on bias prediction
            scores = np.max(((1-prediction), prediction), axis=0)
            rounding = np.round(prediction)

            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_attach_data)
            node_cb = instance_cpx.register_callback(callbacks_cplex.node_selection)

            branch_cb.scoring_function = 'sum'
            branch_cb.scores = scores
            branch_cb.rounding = rounding

            instance_cpx.solve()


    """ Get solving performance statistics """
