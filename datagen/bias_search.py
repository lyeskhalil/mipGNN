import os
import sys
import cplex
from cplex.exceptions import CplexError
import networkx as nx
from networkx.algorithms import bipartite
import random
import time
import numpy as np
import argparse

from pathlib import Path
import glob
import sys
sys.path.extend(["/home/khalile2/projects/def-khalile2/software/DiscreteNet"])
from discretenet.problems.gisp import GISPProblem
from discretenet.problems.fcmnf import FCMNFProblem
import pickle

def disable_output_cpx(instance_cpx):
    instance_cpx.set_log_stream(None)
    # instance_cpx.set_error_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)


def labelVCG(vcg, bias_vector, ip):
    # set atttribute 'bipartite' to 0 for variable nodes, to 1 otherwise
    nx.set_node_attributes(vcg, 1, "bipartite")

    attribute_dict = {}
    bias_dict = {}
    for index, name in enumerate(ip.variables.get_names()):
        name = name.replace('(','[')
        name = name.replace(')',']')
        name = name.replace('_',',')
        attribute_dict[name] = {'bipartite':0, 'bias':bias_vector[index]}
    nx.set_node_attributes(vcg, attribute_dict)


def solveIP(ip, timelimit, mipgap, relgap_pool, maxsols, threads, memlimit, treememlimit, cpx_tmp):
    ip.parameters.emphasis.mip.set(1)
    ip.parameters.threads.set(threads)
    ip.parameters.workmem.set(memlimit)
    ip.parameters.timelimit.set(timelimit)
    # ip.parameters.mip.limits.treememory.set(treememlimit)
    ip.parameters.mip.strategy.file.set(2)
    ip.parameters.workdir.set(cpx_tmp)

    print("Starting Phase I.")
    phase1_time = ip.get_time()
    ip.solve()
    phase1_time = ip.get_time() - phase1_time

    phase1_gap = 1e9
    if ip.solution.is_primal_feasible():
        phase1_gap = ip.solution.MIP.get_mip_relative_gap()
    phase1_status = ip.solution.get_status_string()
    phase2_bestobj = ip.solution.get_objective_value()

    phase2_status, phase2_gap = -1, -1
    print("Finished Phase I.")

    ip.parameters.mip.tolerances.mipgap.set(min([1.0, max([phase1_gap, mipgap])])) #er_200_SET2_1k was with 0.1

    """ https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.SolnPoolInterface-class.html#get_values """
    # 2 = Moderate: generate a larger number of solutions
    ip.parameters.mip.pool.intensity.set(2)
    # Replace the solution which has the worst objective
    ip.parameters.mip.pool.replace.set(1)
    # Maximum number of solutions generated for the solution pool by populate
    ip.parameters.mip.limits.populate.set(maxsols)
    # Maximum pool size
    ip.parameters.mip.pool.capacity.set(maxsols)
    # Relative gap for the solution pool
    ip.parameters.mip.pool.relgap.set(relgap_pool) #er_200_SET2_1k was with 0.2

    phase2_time = ip.get_time()
    try:
        print("Starting Phase II.")
        ip.populate_solution_pool()
        if ip.solution.is_primal_feasible():
            phase2_gap = ip.solution.MIP.get_mip_relative_gap()
            phase2_bestobj = ip.solution.get_objective_value()
            phase2_status = ip.solution.get_status_string()
        print("Finished Phase II.")

    except CplexError as exc:
        phase2_time = 0.0
        print(exc)
        return
    phase2_time = ip.get_time() - phase2_time

    return phase1_status, phase1_gap, phase1_time, phase2_status, phase2_gap, phase2_bestobj, phase2_time


def search(
    mps_path,
    timelimit=120.0,
    threads=1,
    memlimit=2000,
    treememlimit=20000,
    mipgap=0.1,
    relgap_pool=0.1,
    maxsols=1000,
    cpx_output=0,
    cpx_tmp="./cpx_tmp/",
    overwrite=False
):
    
    instance_name_noext = os.path.splitext(mps_path)[0]
    vcg_path = "%s_graph.pkl" % instance_name_noext
    vcg_labeled_path = "%s_graph_bias.pkl" % instance_name_noext
    parameters_path = "%s_parameters.pkl" % instance_name_noext
    results_path = "%s_results.log" % instance_name_noext
    npz_path = "%s_pool.npz" % instance_name_noext

    assert(Path(mps_path).is_file())
    assert(Path(parameters_path).is_file())

    print(mps_path)

    # Create IP, write it to file, and solve it with CPLEX
    ip = cplex.Cplex(mps_path)
    # ip, variable_names = createIP(g, E2, lp_dir + "/" + lpname)
    print("Read in MIP instance.")
    print(ip.variables.get_num(), ip.linear_constraints.get_num(), ip.linear_constraints.get_num_nonzeros())

    # disable all cplex output
    if not cpx_output:
        disable_output_cpx(ip)

    print("Creating VCG...")
    # Get VCG or create it 
    if Path(vcg_path).is_file():
        vcg = nx.read_gpickle(vcg_path)
    else:
        with open(parameters_path, "rb") as fd:
            params  = pickle.load(fd)
        if "gisp" in mps_path:
            loaded_problem = GISPProblem(**params)
        elif "fcmnf" in mps_path:
            loaded_problem = FCMNFProblem(**params)
        vcg = loaded_problem.get_variable_constraint_graph()

    if overwrite or not Path(npz_path).is_file():
        num_solutions = 0
        phase1_gap = None
        phase2_bestobj = None

        start_time = ip.get_time()
        phase1_status, phase1_gap, phase1_time, phase2_status, phase2_gap, phase2_bestobj, phase2_time = solveIP(
            ip,
            timelimit, 
            mipgap, 
            relgap_pool, 
            maxsols, 
            threads, 
            memlimit, 
            treememlimit,
            cpx_tmp)
        end_time = ip.get_time()
        total_time = end_time - start_time

        if not ip.solution.is_primal_feasible():
            print("MIP Infeasible, aborting")
            return

        num_solutions = ip.solution.pool.get_num()
        results_str = ("%s,%s,%g,%s,%g,%g,%d,%g,%g,%g\n" % (
                mps_path, 
                phase1_status, 
                phase1_gap, 
                phase2_status, 
                phase2_gap, 
                phase2_bestobj, 
                num_solutions, 
                total_time,
                phase1_time,
                phase2_time))
        print(results_str)

        with open(results_path, "w+") as results_file:
            results_file.write(results_str)

        if num_solutions >= 1:
            # Collect solutions from pool
            solutions_matrix = np.zeros((num_solutions, len(ip.solution.pool.get_values(0))))
            objval_arr = np.zeros(num_solutions)
            for sol_idx in range(num_solutions):
                sol_objval = ip.solution.pool.get_objective_value(sol_idx)
                objval_arr[sol_idx] = sol_objval 
                solutions_matrix[sol_idx] = ip.solution.pool.get_values(sol_idx)
            solutions_obj_matrix = np.concatenate((np.expand_dims(objval_arr, axis=0).T, solutions_matrix), axis=1)

            with open(npz_path, 'wb') as f:
                np.savez_compressed(f, solutions=solutions_obj_matrix)
            print("Wrote npz file.")

            bias_vector = np.mean(solutions_matrix, axis=0)
    else:
        solutions_matrix = np.load(npz_path)['solutions'][:,1:]
        print("Read existing npz file.")

    bias_vector = np.mean(solutions_matrix, axis=0)

    # Create variable-constraint graph
    labelVCG(vcg, bias_vector, ip)

    nx.write_gpickle(vcg, vcg_labeled_path)
    print("Wrote graph pickle file.")

if __name__ == "__main__":
    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mps_path", type=str)
    parser.add_argument("-timelimit", type=float, default=120.0)
    parser.add_argument("-threads", type=int, default=4)
    parser.add_argument("-memlimit", type=int, default=2000)
    parser.add_argument("-treememlimit", type=int, default=20000)
    parser.add_argument("-mipgap", type=float, default=0.1)
    parser.add_argument("-relgap_pool", type=float, default=0.1)
    parser.add_argument("-maxsols", type=int, default=1000)
    parser.add_argument("-cpx_output", type=int, default=1)
    parser.add_argument("-cpx_tmp", type=str, default="./cpx_tmp/")

    args = parser.parse_args()
    print(args)

    search(
    args.mps_path,
    args.timelimit,
    args.threads,
    args.memlimit,
    args.treememlimit,
    args.mipgap,
    args.relgap_pool,
    args.maxsols,
    args.cpx_output,
    args.cpx_tmp
    )
