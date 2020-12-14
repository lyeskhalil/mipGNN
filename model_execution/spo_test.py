import os
import numpy as np
import argparse
from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import networkx as nx
import cplex
import pickle
import time


if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-instance", type=str)
    parser.add_argument("-model_path", type=str)
    parser.add_argument("-graph", type=str)
    parser.add_argument("-groundtruth", type=str)
    parser.add_argument("-logfile", type=str)
    parser.add_argument("-timelimit", type=float, default=3600)

    parser.add_argument("-method", type=str, default='standard')
    parser.add_argument("-single_model", type=bool, default=False)

    args = parser.parse_args()
    print(args)

    graph = nx.read_gpickle(args.graph)

    # Read true optval to get regret
    with open(args.groundtruth, 'r') as file:
        results_str = file.read()
    objval_true = float(results_str.split(',')[-3])

    if args.method != 'mipgnn':
        # Read model from file
        models = {}
        model_filename = args.model_path
        if args.single_model:
            models[0] = pickle.load(open(model_filename, 'rb'))
        else:
            for entry in os.scandir(args.model_path):
                if entry.name.endswith('.pk'):
                    indicator = int(entry.name[entry.name.rfind('_')+1:len(entry.name)-3])
                    models[indicator] = pickle.load(open(entry, 'rb'))

        # Read MIP 
        instance_cpx = cplex.Cplex(args.instance)
        instance_cpx.parameters.timelimit.set(args.timelimit)

        instance_obj_true = np.array(instance_cpx.objective.get_linear())
        num_variables = len(instance_obj_true)
        if instance_cpx.objective.sense[instance_cpx.objective.get_sense()] == 'maximize':
            instance_cpx.objective.set_sense(instance_cpx.objective.sense.minimize)
            instance_obj_true = -np.array(instance_cpx.objective.get_linear())
            objval_true *= -1

        # print(instance_cpx.objective.get_linear())

        start_time = time.time()
        for node, node_data in graph.nodes(data=True):
            if node_data['bipartite'] == 0:
                indicator = node_data['model_indicator']
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
        regret_unambiguous = np.abs(objval_true - objval_prediction_worstcase)

    else:
        prediction, node_to_varnode = predict.get_prediction(model_name=args.model, graph=graph)
        dict_varname_seqid = predict.get_variable_cpxid(graph, node_to_varnode, prediction)

    print("Ambiguous Regret = ", regret_ambiguous)
    print(objval_true, objval_prediction)
    print("Unambiguous Regret = ", regret_unambiguous)
    print(objval_true, objval_prediction_worstcase)

    results_str = "%.5f,%.5f,%g,%g,%g,%.5f,%.5f,%.5f" % (
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
