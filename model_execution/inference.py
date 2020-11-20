import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
import argparse
import io
import heapq
from pathlib import Path

import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
from gnn_models.mip_bipartite_arch import SimpleNet

import cplex

import callbacks_cplex
import utils

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

def set_cplex_priorities(instance_cpx, prediction):
    # score variables based on bias prediction
    scores = np.max((-(1-prediction), -prediction), axis=0)
    priorities = np.argsort(scores)

    # set priorities
    # reference: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.OrderInterface-class.html
    order_tuples = []
    var_names = instance_cpx.variables.get_names()

    cur_priority = 0
    for priority, var_cpxid in enumerate(priorities):
        var_name = var_names[var_cpxid]
        # print(scores[var_cpxid], scores[priorities[priority-1]])
        if priority > 0 and scores[var_cpxid] > scores[priorities[priority-1]] + 1e-3:
            cur_priority += 1
            # print(cur_priority)
        order_tuples += [(var_name, cur_priority, instance_cpx.order.branch_direction.up)]

    # print(cur_priority)
    # z=1/0
    instance_cpx.order.set(order_tuples)
    # print(instance_cpx.order.get())

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default='default')
    parser.add_argument("-instance", type=str, default='../gisp_generator/LP/er_200_SET2_1k/er_n=200_m=1867_p=0.10_SET2_setparam=100.00_alpha=0.75_606.lp')
    parser.add_argument("-graph", type=str, default='')
    parser.add_argument("-model", type=str, default='../gnn_models/trained_model_er_200_SET2_1k_new')
    parser.add_argument("-barebones", type=int, default=1)
    parser.add_argument("-timelimit", type=float, default=60)
    parser.add_argument("-logfile", type=str, default='sys.stdout')

    # Parameters for exact local branching
    parser.add_argument("-lb_threshold", type=int, default=5)

    args = parser.parse_args()

    instance_path_split = args.instance.split('/')
    instance_name = instance_path_split[-2] + '/' + instance_path_split[-1][:-3]

    # logdir = args.logdir
    # if logdir != 'sys.stdout':
    #     logfile_path = logdir + '/' + instance_path_split[-1][:-3] + '.out'
    #     Path(logdir).mkdir(parents=True, exist_ok=True)

    """ Create CPLEX instance """
    instance_cpx = cplex.Cplex(args.instance)
    # instance_cpx = cplex.Cplex("../gisp_generator/LP/" + instance_name + ".lp")

    """ Set CPLEX parameters, if any """
    instance_cpx.parameters.timelimit.set(args.timelimit)
    instance_cpx.parameters.emphasis.mip.set(1)
    instance_cpx.parameters.mip.display.set(3)
    instance_cpx.parameters.threads.set(1)
    if args.barebones:
        instance_cpx.parameters.mip.limits.cutpasses.set(-1)
        instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
        instance_cpx.parameters.preprocessing.presolve.set(0)

        # DFS = 0, BEST-BOUND = 1 (default), BEST-EST = 2, BEST-EST-ALT = 3
        # instance_cpx.parameters.mip.strategy.nodeselect.set(3)

    """ Solve CPLEX instance with user-selected method """
    if args.method != 'default':
        """ Read in the pickled graph and the trained model """
        # graph = nx.read_gpickle("../gisp_generator/DATA/" + instance_name + ".pk")
        graph = nx.read_gpickle(args.graph)
        prediction, node_to_varnode = get_prediction(model_name=args.model, graph=graph)
        dict_varname_seqid = get_variable_cpxid(graph, node_to_varnode, prediction)
        # print(prediction)
        # todo check dimensions of p

        num_variables = instance_cpx.variables.get_num()
        var_names = instance_cpx.variables.get_names()
        prediction_reord = [dict_varname_seqid[var_name][1] for var_name in var_names]
        prediction = np.array(prediction_reord)

        if 'local_branching' in args.method:
            pred_one_coeff = (prediction >= 0.9) * (-1)
            pred_zero_coeff = (prediction <= 0.1)
            num_ones = -np.sum(pred_one_coeff)
            coeffs = pred_one_coeff + pred_zero_coeff

            local_branching_coeffs = [list(range(len(prediction))), coeffs.tolist()]

        elif args.method == 'branching_priorities':
            set_cplex_priorities(instance_cpx, prediction)

        elif args.method == 'local_branching_approx':
            instance_cpx.linear_constraints.add(
                lin_expr=[local_branching_coeffs],
                senses=['L'],
                rhs=[float(args.lb_threshold - num_ones)],
                names=['local_branching'])
        
        elif args.method == 'local_branching_exact':
            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_local_exact)

            branch_cb.coeffs = local_branching_coeffs
            branch_cb.threshold = args.lb_threshold - num_ones
            branch_cb.is_root = True

        elif args.method == 'node_selection':
            # score variables based on bias prediction
            scores = np.max(((1-prediction), prediction), axis=0)
            rounding = np.round(prediction)

            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_attach_data2)
            node_cb = instance_cpx.register_callback(callbacks_cplex.node_selection3)

            branch_cb.scoring_function = 'sum' #'estimate'
            branch_cb.scores = scores
            branch_cb.rounding = rounding

            node_cb.last_best = 0
            node_cb.freq_best = 10

            node_priority = []
            branch_cb.node_priority = node_priority
            node_cb.node_priority = node_priority            

            branch_cb.time = 0
            node_cb.time = 0

    if args.method == 'default_emptycb':
        branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_empty)

    """ CPLEX output management """
    logstring = sys.stdout
    summary_string = sys.stdout
    if args.logfile != 'sys.stdout':
        logstring = io.StringIO()
        summary_string = io.StringIO()
        instance_cpx.set_log_stream(logstring)
        instance_cpx.set_results_stream(logstring)
        instance_cpx.set_warning_stream(logstring)
        # instance_cpx.set_error_stream(logstring)
        instance_cpx.set_error_stream(open(os.devnull, 'w'))

    # todo: consider runseeds 
    #  https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/refpythoncplex/html/cplex.Cplex-class.html?view=kc#runseeds
    start_time = instance_cpx.get_time()            
    instance_cpx.solve()
    end_time = instance_cpx.get_time()

    """ Get solving performance statistics """
    incumbent_str = ''
    if instance_cpx.solution.is_primal_feasible():
        cplex_status = instance_cpx.solution.get_status_string()
        best_objval = instance_cpx.solution.get_objective_value()
        gap = instance_cpx.solution.MIP.get_mip_relative_gap()
        num_nodes = instance_cpx.solution.progress.get_num_nodes_processed()
        total_time = end_time - start_time

        summary_string.write('solving stats,%s,%g,%g,%g,%i\n' % (
            cplex_status, 
            best_objval,
            gap,
            total_time,
            num_nodes))
    else:
        summary_string.write('solving stats,no solutions found\n')

    if args.logfile != 'sys.stdout':
        if instance_cpx.solution.is_primal_feasible():
            _, incumbent_str = utils.parse_cplex_log(logstring.getvalue())
            summary_string.write(incumbent_str)
        summary_string = summary_string.getvalue()
        with open(args.logfile, 'w') as logfile:
            logfile.write(summary_string)
