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
from sklearn.datasets import make_regression 
from sklearn.preprocessing import PolynomialFeatures


def disable_output_cpx(instance_cpx):
    instance_cpx.set_log_stream(None)
    # instance_cpx.set_error_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)

def dimacsToNx(filename):
    g = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            arr = line.split()
            if line[0] == 'e':
                g.add_edge(int(arr[1]), int(arr[2]))
    return g

def generateRevsCosts(g, whichSet, setParam):
    if whichSet == 'SET1':
        for node in g.nodes():
            g.nodes[node]['revenue'] = random.randint(1,100)
        for u,v,edge in g.edges(data=True):
            edge['cost'] = np.round((g.nodes[u]['revenue'] + g.nodes[v]['revenue'])/float(setParam))
    elif whichSet == 'SET2':
        for node in g.nodes():
            g.nodes[node]['revenue'] = float(setParam)
        for u,v,edge in g.edges(data=True):
            edge['cost'] = 1.0

def generateDataPolySPO(true_func, num_data, poly_deg=1, n_features=10, bias=0, noise_halfwidth=0.5, a_min=None, a_max=None):
    feature_matrix = np.random.normal(loc=0, scale=1, size=(num_data, n_features))

    output_vector = np.expand_dims(np.power(feature_matrix.dot(true_func), poly_deg), axis=1)
    output_vector += bias
    output_vector *= np.random.rand(num_data,1) * (2 * noise_halfwidth) + (1 - noise_halfwidth)

    output_vector = np.clip(output_vector, a_min=a_min, a_max=a_max)

    return feature_matrix, output_vector

def generateDataSPO(true_func, g, E2, n_features=10, nodes_feature_factor=100, nodes_dummy=-5, noise_loc=0, noise_scale=50, bias=1000):
    num_nodes = nx.number_of_nodes(g)
    num_edges = len(E2)

    true_func = np.expand_dims(np.append(true_func, [bias]), axis=1)

    print(true_func)

    feature_matrix = np.random.rand(num_nodes+num_edges, n_features+1)
    feature_matrix[:num_nodes,:] *= nodes_feature_factor
    feature_matrix[:num_nodes,-1] = nodes_dummy
    feature_matrix[num_nodes:,-1] = 0

    output_vector = feature_matrix.dot(true_func) 
    output_vector += np.random.normal(loc=noise_loc, scale=noise_scale, size=(num_nodes+num_edges,1))
    output_vector[:num_nodes] = np.clip(output_vector[:num_nodes], a_min=None, a_max=0)
    output_vector[num_nodes:] = np.clip(output_vector[num_nodes:], a_min=1.0, a_max=None)

    return feature_matrix, output_vector

def generateRevsCostsSPO(g, E2, feature_matrix, output_vector):
    num_nodes = nx.number_of_nodes(g)
    counter = 0
    for node in g.nodes():
        g.nodes[node]['features'] = feature_matrix[counter,:].tolist()
        g.nodes[node]['revenue'] = float(-output_vector[counter])
        g.nodes[node]['objcoeff'] = float(output_vector[counter])
        g.nodes[node]['model_indicator'] = 0
        
        counter += 1

        if counter == 1:
            print(g.nodes[node]['features'], output_vector[counter])

    for u,v,edge in g.edges(data=True):
        if edge['E2']:
            edge['features'] = feature_matrix[counter,:].tolist()
            edge['cost'] = float(output_vector[counter])
            edge['objcoeff'] = float(output_vector[counter])
            edge['model_indicator'] = 1

            counter += 1

            if counter == num_nodes + 1:
                print(edge['features'], output_vector[counter])

def generateRevsCostsSPO_sameNodesEdges(g, E2, n_features=10, n_informative=10, bias=1000):
    num_nodes = nx.number_of_nodes(g)
    num_edges = len(E2)

    rng = np.random.RandomState(0)
    _, _, true_func = make_regression(
        n_samples=1,
        n_features=n_features,
        n_informative=n_features,
        coef=True,
        random_state=rng)

    true_func = np.expand_dims(np.append(true_func, [bias]), axis=1)

    print(true_func)

    feature_matrix = np.random.rand(num_nodes+num_edges, n_features+1)
    feature_matrix[:num_nodes,-1] = -1
    feature_matrix[num_nodes:,-1] = 1
    output_vector = feature_matrix.dot(true_func) + np.random.normal(loc=10, scale=2, size=(num_nodes+num_edges,1))

    counter = 0
    for node in g.nodes():
        g.nodes[node]['features'] = feature_matrix[counter,:].tolist()
        g.nodes[node]['revenue'] = float(-output_vector[counter])
        g.nodes[node]['objcoeff'] = float(output_vector[counter])
        counter += 1
    for u,v,edge in g.edges(data=True):
        if edge['E2']:
            edge['features'] = feature_matrix[counter,:].tolist()
            edge['cost'] = float(output_vector[counter])
            edge['objcoeff'] = float(output_vector[counter])
            counter += 1

    return feature_matrix, output_vector

def generateE2(g, alphaE2):
    E2 = set()
    for u,v,edge in g.edges(data=True):
        if random.random() <= alphaE2:
            E2.add((u,v))
            edge['E2'] = True
        else:
            edge['E2'] = False
    return E2

def createIP(g, E2, ipfilename):  
    n = nx.number_of_nodes(g)
    m = nx.number_of_edges(g)

    objective_coeffs = []
    variable_types = "B"*(n+len(E2))
    constraint_rhs = [1]*(m)
    constraint_senses = "L"*(m)
    
    variable_names = []
    for node in g.nodes():
        variable_names.append("x" + str(node))
        objective_coeffs.append(g.nodes[node]['revenue'])
    for edge in E2:
        variable_names.append("y" + str(edge[0]) + "_" + str(edge[1]))
        objective_coeffs.append(-1*g[edge[0]][edge[1]]['cost'])
            
    rows = []
    for node1, node2, edge in g.edges(data=True):
        if (node1,node2) in E2:
            edge_varname = "y" + str(node1) + "_" + str(node2)
            row = [[edge_varname, "x" + str(node1), "x" + str(node2)],[-1,1,1]]
        else:
            row = [["x" + str(node1), "x" + str(node2)],[1,1]]
        rows.append(row)

    ip = cplex.Cplex()
    ip.set_problem_name(ipfilename)
    ip.objective.set_sense(ip.objective.sense.maximize)
    ip.variables.add(obj=objective_coeffs, types=variable_types, names=variable_names)
    ip.linear_constraints.add(lin_expr=rows, senses=constraint_senses, rhs=constraint_rhs)
    
    ip.write(ipfilename + '.lp')
    return ip, variable_names

def extractVCG(g, E2, ip, set_biases, spo, gap=None, bestobj=None):
    num_solutions = 0
    if set_biases:
        num_solutions = ip.solution.pool.get_num()
        bias_arr = np.zeros(len(ip.solution.pool.get_values(0)))

        for sol_idx in range(ip.solution.pool.get_num()):
            # if ip.solution.pool.get_objective_value(sol_idx) <= 0:
            #     num_solutions -= 1
            bias_arr += ip.solution.pool.get_values(sol_idx)
        bias_arr /= num_solutions

        bias_dict = {}
        for index, name in enumerate(ip.variables.get_names()):
            bias_dict[name] = bias_arr[index]

        print("num_solutions = %d" % num_solutions)

    vcg = nx.Graph(num_solutions=num_solutions, gap=gap, bestobj=bestobj)

    vcg.add_nodes_from([("x" + str(node), {'objcoeff':-node_data['revenue']}) for node, node_data in g.nodes(data=True)], bipartite=0)
    vcg.add_nodes_from(["y" + str(edge[0]) + "_" + str(edge[1]) for edge in E2], bipartite=0)

    for node, node_data in g.nodes(data=True):
        node_name = "x" + str(node)
        bias = bias_dict[node_name] if set_biases else 0
        vcg.add_node(node_name, bias=bias, objcoeff=-1*node_data['revenue'], bipartite=0)

        if spo:
            vcg.nodes[node_name]['features'] = node_data['features']

    for edge in E2:
        node_name = "y" + str(edge[0]) + "_" + str(edge[1])
        bias = bias_dict[node_name] if set_biases else 0
        vcg.add_node(node_name, bias=bias, objcoeff=g[edge[0]][edge[1]]['cost'], bipartite=0)

        if spo:
            vcg.nodes[node_name]['features'] = g[edge[0]][edge[1]]['features']
    
    constraint_counter = 0        
    for node1, node2, edge in g.edges(data=True):
        node_name = "c" + str(constraint_counter)
        vcg.add_node(node_name, rhs=1.0, bipartite=1)
        if (node1,node2) in E2:
            edge_varname = "y" + str(node1) + "_" + str(node2)
            vcg.add_edge(edge_varname, node_name, coeff=-1)
        vcg.add_edge("x" + str(node1), node_name, coeff=1)
        vcg.add_edge("x" + str(node2), node_name, coeff=1)
        constraint_counter += 1

    return vcg


def solveIP(ip, pool_bool, timelimit, mipgap, relgap_pool, maxsols, threads, memlimit, treememlimit, cpx_tmp):
    ip.parameters.emphasis.mip.set(1)
    ip.parameters.threads.set(threads)
    ip.parameters.workmem.set(memlimit)
    ip.parameters.timelimit.set(timelimit)
    # ip.parameters.mip.limits.treememory.set(treememlimit)
    ip.parameters.mip.strategy.file.set(2)
    ip.parameters.workdir.set(cpx_tmp)
    
    ip.solve()

    phase1_gap = 1e9
    if ip.solution.is_primal_feasible():
        phase1_gap = ip.solution.MIP.get_mip_relative_gap()
    phase1_status = ip.solution.get_status_string()
    phase2_bestobj = ip.solution.get_objective_value()

    phase2_status, phase2_gap = -1, -1
    if pool_bool:
        print("Finished Phase I.")

        ip.parameters.mip.tolerances.mipgap.set(max([phase1_gap, mipgap])) #er_200_SET2_1k was with 0.1

        """ https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.SolnPoolInterface-class.html#get_values """
        # 2 = Moderate: generate a larger number of solutions
        ip.parameters.mip.pool.intensity.set(2)
        # Replace the solution which has the worst objective
        ip.parameters.mip.pool.replace.set(1)
        # Maximum number of solutions generated for the solution pool by populate
        ip.parameters.mip.limits.populate.set(maxsols)
        # Relative gap for the solution pool
        ip.parameters.mip.pool.relgap.set(relgap_pool) #er_200_SET2_1k was with 0.2

        try:
            ip.populate_solution_pool()
            if ip.solution.is_primal_feasible():
                phase2_gap = ip.solution.MIP.get_mip_relative_gap()
                phase2_bestobj = ip.solution.get_objective_value()
                phase2_status = ip.solution.get_status_string()
            print("Finished Phase II.")

        except CplexError as exc:
            print(exc)
            return

    return phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj

if __name__ == "__main__":
    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir", type=str)
    parser.add_argument("-instance", type=str, default='')
    parser.add_argument("-min_n", type=int)
    parser.add_argument("-max_n", type=int)
    parser.add_argument("-er_prob", type=float, default=0.1)
    parser.add_argument("-whichSet", type=str, default='SET2')
    parser.add_argument("-setParam", type=float, default=100.0)
    parser.add_argument("-alphaE2", type=float, default=0.75)
    parser.add_argument("-timelimit", type=float, default=120.0)
    parser.add_argument("-solve", type=int, default=1)
    parser.add_argument("-threads", type=int, default=4)
    parser.add_argument("-memlimit", type=int, default=2000)
    parser.add_argument("-treememlimit", type=int, default=20000)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-mipgap", type=float, default=0.1)
    parser.add_argument("-relgap_pool", type=float, default=0.1)
    parser.add_argument("-maxsols", type=int, default=1000)
    parser.add_argument("-overwrite_data", type=int, default=0)
    parser.add_argument("-cpx_output", type=int, default=0)
    parser.add_argument("-cpx_tmp", type=str, default="./tmp/")

    parser.add_argument("-spo", type=int, default=0)    
    parser.add_argument("-spo_halfwidth", type=float, default=0.5)    
    parser.add_argument("-spo_polydeg", type=int, default=2)    
    parser.add_argument("-spo_bias_nodes", type=float, default=-100)    
    parser.add_argument("-spo_bias_edges", type=float, default=10)    

    args = parser.parse_args()
    print(args)

    assert(args.max_n >= args.min_n)

    lp_dir = "LP/" + args.exp_dir
    try: 
        os.makedirs(lp_dir)
    except OSError:
        if not os.path.exists(lp_dir):
            raise

    sol_dir = "SOL/" + args.exp_dir
    try: 
        os.makedirs(sol_dir)
    except OSError:
        if not os.path.exists(sol_dir):
            raise

    data_dir = "DATA/" + args.exp_dir
    try: 
        os.makedirs(data_dir)
    except OSError:
        if not os.path.exists(data_dir):
            raise

    if args.spo:
        spodata_dir = "SPO_DATA/" + args.exp_dir
        try: 
            os.makedirs(spodata_dir)
        except OSError:
            if not os.path.exists(spodata_dir):
                raise

    # Seed generator
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.instance == '':
        # Generate random graph
        numnodes = random.randint(args.min_n, args.max_n+1)
        g = nx.erdos_renyi_graph(n=numnodes, p=args.er_prob, seed=args.seed)
        lpname = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f_%d" % (numnodes, nx.number_of_edges(g), args.er_prob, args.whichSet, args.setParam, args.alphaE2, args.seed))
    else:
        g = dimacsToNx(args.instance)
        # instanceName = os.path.splitext(instance)[1]
        instanceName = args.instance.split('/')[1]
        lpname = ("%s_%s_%g_%g_%d" % (instanceName, args.whichSet, args.alphaE2, args.setParam, args.seed))

    data_fullpath = data_dir + "/" + lpname + ".pk"
    if not args.overwrite_data and os.path.isfile(data_fullpath):
        print("data exists")
        exit() 
    
    # Generate node revenues and edge costs
    generateRevsCosts(g, args.whichSet, args.setParam)

    # Generate the set of removable edges
    E2 = generateE2(g, args.alphaE2)

    if args.spo:
        n_features = 10
                
        np.random.seed(0)
        true_func_nodes = np.random.rand(n_features)# * 10
        true_func_edges = np.random.rand(n_features)# * 10
        print("true_func_nodes = ", true_func_nodes)
        print("true_func_edges = ", true_func_edges)
        np.random.seed(args.seed)
        # feature_matrix, output_vector = generateDataSPO(true_func, g, E2, n_features=n_features, nodes_feature_factor=100, nodes_dummy=-5, noise_loc=0, noise_scale=50, bias=1000)
        
        num_nodes = nx.number_of_nodes(g)
        num_edges = len(E2)
        num_data = num_nodes + num_edges
        feature_matrix1, output_vector1 = generateDataPolySPO(true_func_nodes, num_nodes, n_features=10, poly_deg=args.spo_polydeg, bias=args.spo_bias_nodes, noise_halfwidth=args.spo_halfwidth, a_min=None, a_max=0)
        feature_matrix2, output_vector2 = generateDataPolySPO(true_func_edges, num_edges, n_features=10, poly_deg=args.spo_polydeg, bias=args.spo_bias_edges, noise_halfwidth=args.spo_halfwidth, a_min=0, a_max=None)
        feature_matrix = np.append(feature_matrix1, feature_matrix2, axis=0)
        output_vector = np.append(output_vector1, output_vector2, axis=0)

        print(np.mean(feature_matrix1), np.std(feature_matrix1), np.median(feature_matrix1), np.min(feature_matrix1), np.max(feature_matrix1))
        print(np.mean(feature_matrix2), np.std(feature_matrix2), np.median(feature_matrix2), np.min(feature_matrix2), np.max(feature_matrix2))
        print(np.mean(output_vector1), np.std(output_vector1), np.median(output_vector1), np.min(output_vector1), np.max(output_vector1))
        print(np.mean(output_vector2), np.std(output_vector2), np.median(output_vector2), np.min(output_vector2), np.max(output_vector2))

        generateRevsCostsSPO(g, E2, feature_matrix, output_vector)

        spodata_fullpath = spodata_dir + "/" + lpname + ".csv"
        model_indicator = np.zeros((num_data, 1))
        model_indicator[num_nodes:] = 1
        np.savetxt(spodata_fullpath, np.concatenate((model_indicator, feature_matrix, output_vector), axis=1), delimiter=',')

    # Create IP, write it to file, and solve it with CPLEX
    print(lpname)
    ip, variable_names = createIP(g, E2, lp_dir + "/" + lpname)
    print("Created MIP instance.")

    # disable all cplex output
    if not args.cpx_output:
        disable_output_cpx(ip)

    num_solutions = 0
    phase1_gap = None
    phase2_bestobj = None

    pool_bool = (args.solve == 1)
    if args.solve > 0:
        start_time = ip.get_time()
        phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj = solveIP(
            ip,
            pool_bool, 
            args.timelimit, 
            args.mipgap, 
            args.relgap_pool, 
            args.maxsols, 
            args.threads, 
            args.memlimit, 
            args.treememlimit,
            args.cpx_tmp)
        end_time = ip.get_time()
        total_time = end_time - start_time
 
        num_solutions = ip.solution.pool.get_num()
        results_str = ("%s,%s,%g,%s,%g,%g,%d,%g\n" % (
                lpname, 
                phase1_status, 
                phase1_gap, 
                phase2_status, 
                phase2_gap, 
                phase2_bestobj, 
                num_solutions, 
                total_time))
        print(results_str)

        with open(sol_dir + "/" + lpname + ".sol", "w+") as sol_file:
            sol_file.write(results_str)

        if pool_bool and num_solutions >= 1:
            # Collect solutions from pool
            solutions_matrix = np.zeros((num_solutions, len(ip.solution.pool.get_values(0))))
            objval_arr = np.zeros(num_solutions)
            for sol_idx in range(num_solutions):
                sol_objval = ip.solution.pool.get_objective_value(sol_idx)
                objval_arr[sol_idx] = sol_objval 
                # if sol_objval > 0:
                solutions_matrix[sol_idx] = ip.solution.pool.get_values(sol_idx)
            solutions_obj_matrix = np.concatenate((np.expand_dims(objval_arr, axis=0).T, solutions_matrix), axis=1)

            with open(sol_dir + "/" + lpname + ".npz", 'wb') as f:
                np.savez_compressed(f, solutions=solutions_obj_matrix)
            print("Wrote npz file.")

    # Create variable-constraint graph
    vcg = extractVCG(g, E2, ip, spo=args.spo, set_biases=(args.solve == 1 and num_solutions >= 1), gap=phase1_gap, bestobj=phase2_bestobj)

    nx.write_gpickle(vcg, data_dir + "/" + lpname + ".pk")
    print("Wrote graph pickle file.")

    vcg2=nx.read_gpickle(data_dir + "/" + lpname + ".pk")
    print("Read back graph pickle file.")
    # print(["(%s, %g)" % (n, d['bias']) for n, d in vcg2.nodes(data=True) if d['bipartite']==0])

    print([d['objcoeff'] for n, d in vcg2.nodes(data=True) if d['bipartite']==0])

    if args.solve == 2:
        sol = ip.solution.get_values()
        print(np.sum(sol[:nx.number_of_nodes(g)]), np.sum(sol[:nx.number_of_edges(g)]))
    print(lpname)

