import os
import numpy as np
import argparse
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import networkx as nx
import cplex
import pickle
import time
import re
import torch

import spo_utils
from spo_torch import SPONet, SPOLoss

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-instance", type=str)
    parser.add_argument("-model_dir", type=str)
    parser.add_argument("-model_prefix", type=str)
    parser.add_argument("-graph", type=str)
    parser.add_argument("-groundtruth", type=str)
    parser.add_argument("-logfile", type=str)

    parser.add_argument("-method", type=str, default='standard')
    parser.add_argument("-single_model", type=bool, default=False)

    parser.add_argument("-nn_cpx_timelimit", type=float, default=3600)
    parser.add_argument("-nn_cpx_threads", type=int, default=8)

    args = parser.parse_args()
    print(args)

    graph = nx.read_gpickle(args.graph)

    # Read true optval to get regret
    objval_true = spo_utils.read_optval(args.groundtruth)

    torch_bool = False
    if args.method != 'mipgnn':
        if not args.model_dir.endswith('.pt'):
            model_filename = args.model_dir + '/' + args.model_prefix
            models = spo_utils.read_sklearn_model(args.model_dir, args.model_prefix, args.single_model)
        else:
            model_filename = args.model_dir
            torch_bool = True
            checkpoint = torch.load(model_filename)
            nn_poly_degree = checkpoint['nn_poly_degree']
            if nn_poly_degree > 1:
                poly = PolynomialFeatures(nn_poly_degree, include_bias=False)

            num_features = checkpoint['num_features']
            ints_in_filename = re.findall('\d+',model_filename)#map(int, re.findall(r'\d+', model_filename))
            depth, width = int(ints_in_filename[0]), int(ints_in_filename[1])
            models = [SPONet(num_features, depth, width, relu_sign=-1), SPONet(num_features, depth, width, relu_sign=1)]
            models[0].load_state_dict(checkpoint['model0_state_dict'])
            models[1].load_state_dict(checkpoint['model1_state_dict'])

        # Read MIP 
        instance_cpx = cplex.Cplex(args.instance)
        instance_cpx.parameters.timelimit.set(args.nn_cpx_timelimit)
        instance_cpx.parameters.threads.set(args.nn_cpx_threads)

        instance_obj_true = np.array(instance_cpx.objective.get_linear())
        num_variables = len(instance_obj_true)
        if instance_cpx.objective.sense[instance_cpx.objective.get_sense()] == 'maximize':
            instance_cpx.objective.set_sense(instance_cpx.objective.sense.minimize)
            instance_obj_true *= -1
            objval_true *= -1

        # print(instance_cpx.objective.get_linear())

        start_time = time.time()
        for node, node_data in graph.nodes(data=True):
            if node_data['bipartite'] == 0:
                indicator = node_data['model_indicator']
                if torch_bool:
                    features = node_data['features']
                    if nn_poly_degree > 1:
                        features = poly.fit_transform([features])
                    prediction = float(models[indicator](torch.tensor(features, dtype=models[0].layers[0].weight.dtype)))
                else:
                    prediction = models[indicator].predict([node_data['features']])[0]
                instance_cpx.objective.set_linear(node, prediction)
        time_predictions = time.time() - start_time

        # print(instance_cpx.objective.get_linear())      
        start_time = instance_cpx.get_time()
        instance_cpx.solve()
        time_solve_prediction = instance_cpx.get_time() - start_time

        solution_prediction = np.array(instance_cpx.solution.get_values())
        objval_prediction = solution_prediction.dot(instance_obj_true)
        instance_obj_prediction = np.array(instance_cpx.objective.get_linear())
        objval_prediction_predictedobj = instance_cpx.solution.get_objective_value()

        regret_ambiguous = np.abs(objval_true - objval_prediction)

        # To get unambiguous SPO loss:
        # 1- reset objective of instance_cpx to instance_obj_true
        #    turn problem into maximization
        instance_cpx.objective.set_sense(instance_cpx.objective.sense.maximize)
        instance_cpx.objective.set_linear([(idx, instance_obj_true[idx]) for idx in range(num_variables)])

        # 2- add constraint that predicted obj value isn't worsened
        instance_cpx.linear_constraints.add(
            lin_expr=[[[idx for idx in range(num_variables)], [instance_obj_prediction[idx] for idx in range(num_variables)]]],
            senses=['L'],
            rhs=[objval_prediction_predictedobj + 1e-6])

        # 3- warm start with opt of instance_cpx
        instance_cpx.MIP_starts.add(
            [(idx, int(solution_prediction[idx])) for idx in range(num_variables)],
            instance_cpx.MIP_starts.effort_level.check_feasibility)

        # 4- solve worst case problem and retrieve value
        start_time = instance_cpx.get_time()
        instance_cpx.solve()
        time_solve_loss = instance_cpx.get_time() - start_time

        objval_prediction_worstcase = instance_cpx.solution.get_objective_value()
        # todo: take into account suboptimality of objval_true e.g. timeout
        regret_unambiguous = np.abs(objval_true - objval_prediction_worstcase)

    else:
        prediction, node_to_varnode = predict.get_prediction(model_name=args.model, graph=graph)
        dict_varname_seqid = predict.get_variable_cpxid(graph, node_to_varnode, prediction)

    print("Ambiguous Regret = ", regret_ambiguous)
    print(objval_true, objval_prediction)
    print("Unambiguous Regret = ", regret_unambiguous)
    print(objval_true, objval_prediction_worstcase)

    results_str = "%s,%.5f,%.5f,%g,%g,%g,%.5f,%.5f,%.5f" % (
        model_filename,
        regret_unambiguous,
        regret_ambiguous,
        time_predictions,
        time_solve_loss,
        time_solve_prediction,
        objval_true,
        objval_prediction,
        objval_prediction_worstcase)
    print(results_str)

    with open(args.logfile, "w+") as results_file:
        results_file.write(results_str)
