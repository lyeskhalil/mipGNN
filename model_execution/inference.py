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
from gnn_models.baselines.mpnn_architecture_class import Net

import cplex

import callbacks_cplex

def get_prediction(model_name, graph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(dim=64).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))


    data, var_node, node_var = create_data_object(graph)
    model.eval()

    data = data.to(device)

    # out = model(data).max(dim=1)[1].cpu().detach().numpy()
    out = model(data).exp()[:,1].cpu().detach().numpy()

    return out, node_var, var_node

def create_data_object(graph):
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

def set_cplex_priorities(instance_cpx, p):
    # score variables based on bias prediction
    scores = np.max((-(1-p), -p), axis=0)
    priorities = np.argsort(scores)

    # set priorities
    # reference: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.OrderInterface-class.html
    order_tuples = []
    node_names_list = list(graph.nodes())
    for priority, var_idx_model in enumerate(priorities):
        var_idx = node_var[var_idx_model]
        var_name = node_names_list[var_idx]
        # print(var_name, priority, scores[var_idx_model])
        order_tuples += [(var_name, priority, instance_cpx.order.branch_direction.up)]
        # order_tuples += [(var_name, np.random.randint(num_variables), instance_cpx.order.branch_direction.up)]
        # order_tuples += [(var_name, 1, instance_cpx.order.branch_direction.up)]
    instance_cpx.order.set(order_tuples)
    # print(instance_cpx.order.get())

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default='default')
    parser.add_argument("-instance", type=str, default='er_200_SET2_1k/er_n=200_m=1867_p=0.10_SET2_setparam=100.00_alpha=0.75_606')
    parser.add_argument("-model", type=str, default='../gnn_models/baselines/trained_model_er_200_SET2_1k_SIMPLE')
    args = parser.parse_args()

    instance_name = args.instance

    """ Create CPLEX instance """
    instance_cpx = cplex.Cplex("../gisp_generator/LP/" + instance_name + ".lp")

    """ Set CPLEX parameters, if any """
    # instance_cpx.parameters.mip.strategy.nodeselect.set(0)
    instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
    # instance_cpx.parameters.mip.strategy.variableselect.set(2)
    # instance_cpx.parameters.mip.strategy.search.set(1)
    """ MIP branching direction parameter: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/BrDir.html"""
    # instance_cpx.parameters.mip.strategy.branch.set(1)

    """ CPLEX output management """
    instance_cpx.set_log_stream(sys.stdout)
    instance_cpx.set_results_stream(sys.stdout)

    """ Solve CPLEX instance with user-selected method """
    if args.method == 'default':
        instance_cpx.solve()

    else:
        """ Read in the pickled graph and the trained model """
        graph = nx.read_gpickle("../gisp_generator/DATA/" + instance_name + ".pk")
        p, node_var, var_node = get_prediction(model_name=args.model, graph=graph)
        print(p)
        # todo check dimensions of p

        num_variables = instance_cpx.variables.get_num()

        if args.method == 'branching_priorities':
            set_cplex_priorities(instance_cpx, p)
            instance_cpx.solve()

        elif args.method == 'local_branching_approx':
            pass
        
        elif args.method == 'local_branching_exact':
            pass

        elif args.method == 'node_selection':
            # score variables based on bias prediction
            scores = np.max(((1-p), p), axis=0)
            rounding = np.round(p)

            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_attach_data)
            node_cb = instance_cpx.register_callback(callbacks_cplex.node_selection)

            branch_cb.scores = scores
            branch_cb.rounding = rounding

            instance_cpx.solve()


    """ Get solving performance statistics """
