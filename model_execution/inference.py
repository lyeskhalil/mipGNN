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
from gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet

import cplex

import callbacks_cplex
import utils
import predict


def rename_variables(var_names):
    for i in range(len(var_names)):
        name = var_names[i]
        name = name.replace('(','[')
        name = name.replace(')',']')
        name = name.replace('_',',')
        var_names[i] = name
    return var_names


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
        # if priority > 0 and scores[var_cpxid] > scores[priorities[priority-1]] + 1e-3:
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

    # Parameters for primal heuristic mip start
    parser.add_argument("-rounding_threshold", type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    instance_path_split = args.instance.split('/')
    instance_name = instance_path_split[-2] + '/' + instance_path_split[-1][:-3]

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
    if 'default' not in args.method:
        """ Read in the pickled graph and the trained model """
        # graph = nx.read_gpickle("../gisp_generator/DATA/" + instance_name + ".pk")
        graph = nx.read_gpickle(args.graph)
        prediction, node_to_varnode = predict.get_prediction(model_name=args.model, graph=graph)
        dict_varname_seqid = predict.get_variable_cpxid(graph, node_to_varnode, prediction)
        # print(prediction)
        # todo check dimensions of p

        num_variables = instance_cpx.variables.get_num()
        var_names = rename_variables(instance_cpx.variables.get_names())
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

        elif 'node_selection' in args.method:
            # score variables based on bias prediction
            scores = np.max(((1-prediction), prediction), axis=0)
            rounding = np.round(prediction)

            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_attach_data2)
            node_cb = instance_cpx.register_callback(callbacks_cplex.node_selection3)

            branch_cb.scoring_function = 'sum' #'estimate'
            branch_cb.scores = scores
            branch_cb.rounding = rounding

            node_cb.last_best = 0
            node_cb.freq_best = 100

            node_priority = []
            branch_cb.node_priority = node_priority
            node_cb.node_priority = node_priority            

            branch_cb.time = 0
            node_cb.time = 0

            if 'branching' in args.method:
                set_cplex_priorities(instance_cpx, prediction)

        elif args.method == 'primal_mipstart':
            # instance_cpx.parameters.mip.limits.nodes.set(1)

            # threshold_set = np.minimum(prediction, 1-prediction)
            # threshold_set = np.sort(np.unique(threshold_set))

            threshold_set = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5]

            # threshold = args.rounding_threshold
            for threshold in threshold_set:
                indices_integer = np.where((prediction >= 1-threshold) | (prediction <= threshold))[0]
                print(len(indices_integer), len(prediction))

                instance_cpx.MIP_starts.add(
                    cplex.SparsePair(
                        ind=indices_integer.tolist(),
                        val=np.round(prediction[indices_integer]).tolist()),
                    instance_cpx.MIP_starts.effort_level.repair)

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
    # if 'primal_' not in args.method:            
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
