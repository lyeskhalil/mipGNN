import os
import sys
import cplex
from cplex.exceptions import CplexError
import networkx as nx
from networkx.algorithms import bipartite
import random
import time
import numpy as np

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

def extractVCG(g, E2, ip):
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

    vcg = nx.Graph(num_solutions=num_solutions)

    vcg.add_nodes_from([("x" + str(node), {'objcoeff':-node_data['revenue']}) for node, node_data in g.nodes(data=True)], bipartite=0)
    vcg.add_nodes_from(["y" + str(edge[0]) + "_" + str(edge[1]) for edge in E2], bipartite=0)

    for node, node_data in g.nodes(data=True):
        node_name = "x" + str(node)
        bias = bias_dict[node_name]
        # for sol_idx in range(num_solutions):
        #     bias += ip.solution.pool.get_values(sol_idx, node_name) / num_solutions
        vcg.add_node(node_name, bias=bias, objcoeff=-1*node_data['revenue'], bipartite=0)
    for edge in E2:
        node_name = "y" + str(edge[0]) + "_" + str(edge[1])
        bias = bias_dict[node_name]
        # for sol_idx in range(num_solutions):
        #     bias += ip.solution.pool.get_values(sol_idx, node_name) / num_solutions
        vcg.add_node(node_name, bias=bias, objcoeff=g[edge[0]][edge[1]]['cost'], bipartite=0)
    
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


def solveIP(ip, timelimit, mipgap, relgap_pool, maxsols, threads):
    ip.parameters.threads.set(threads)

    ip.parameters.timelimit.set(timelimit)
    
    ip.solve()

    phase1_gap = 1e9
    if ip.solution.is_primal_feasible():
        phase1_gap = ip.solution.MIP.get_mip_relative_gap()
    phase1_status = ip.solution.get_status_string()

    ip.parameters.mip.tolerances.mipgap.set(mipgap) #er_200_SET2_1k was with 0.1

    """ https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.SolnPoolInterface-class.html#get_values """
    # 2 = Moderate: generate a larger number of solutions
    ip.parameters.mip.pool.intensity.set(2)
    # Replace the solution which has the worst objective
    ip.parameters.mip.pool.replace.set(1)
    # Maximum number of solutions generated for the solution pool by populate
    ip.parameters.mip.limits.populate.set(maxsols)
    # Relative gap for the solution pool
    ip.parameters.mip.pool.relgap.set(relgap_pool) #er_200_SET2_1k was with 0.2

    # disable all cplex output
#     ip.set_log_stream(None)
#     ip.set_error_stream(None)
#     ip.set_warning_stream(None)
#     ip.set_results_stream(None)

    try:
        ip.populate_solution_pool()
        if ip.solution.is_primal_feasible():
            phase2_gap = ip.solution.MIP.get_mip_relative_gap()
            phase2_bestobj = ip.solution.get_objective_value()
            phase2_status = ip.solution.get_status_string()

    except CplexError as exc:
        print(exc)
        return

    return phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj

if __name__ == "__main__":
    instance = None
    exp_dir = None
    min_n = None
    max_n = None
    er_prob = 0.1
    whichSet = 'SET2'
    setParam = 100.0
    alphaE2 = 0.75
    timelimit = 120.0
    solveInstance = True

    threads = 4
    mipgap = 0.1
    relgap_pool = 0.1
    maxsols = 1000

    seed = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-instance':
            instance = sys.argv[i + 1]
        if sys.argv[i] == '-exp_dir':
            exp_dir = sys.argv[i + 1]
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-er_prob':
            er_prob = float(sys.argv[i + 1])
        if sys.argv[i] == '-whichSet':
            whichSet = sys.argv[i + 1]
        if sys.argv[i] == '-setParam':
            setParam = float(sys.argv[i + 1])
        if sys.argv[i] == '-alphaE2':
            alphaE2 = float(sys.argv[i + 1])
        if sys.argv[i] == '-timelimit':
            timelimit = float(sys.argv[i + 1])
        if sys.argv[i] == '-solve':
            solveInstance = bool(sys.argv[i + 1])
        if sys.argv[i] == '-seed':
            seed = int(sys.argv[i + 1])
    assert exp_dir is not None
    if instance is None:
        assert min_n is not None
        assert max_n is not None

    lp_dir = "LP/" + exp_dir
    try: 
        os.makedirs(lp_dir)
    except OSError:
        if not os.path.exists(lp_dir):
            raise

    sol_dir = "SOL/" + exp_dir
    try: 
        os.makedirs(sol_dir)
    except OSError:
        if not os.path.exists(sol_dir):
            raise

    data_dir = "DATA/" + exp_dir
    try: 
        os.makedirs(data_dir)
    except OSError:
        if not os.path.exists(data_dir):
            raise
        
    # Seed generator
    random.seed(seed)

    print(whichSet)
    print(setParam)
    print(alphaE2)
    
    if instance is None:
        # Generate random graph
        numnodes = random.randint(min_n, max_n+1)
        g = nx.erdos_renyi_graph(n=numnodes, p=er_prob, seed=seed)
        lpname = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f_%d" % (numnodes, nx.number_of_edges(g), er_prob, whichSet, setParam, alphaE2, seed))
    else:
        g = dimacsToNx(instance)
        # instanceName = os.path.splitext(instance)[1]
        instanceName = instance.split('/')[1]
        lpname = ("%s_%s_%g_%g_%d" % (instanceName, whichSet, alphaE2, setParam, seed))
        
    # Generate node revenues and edge costs
    generateRevsCosts(g, whichSet, setParam)
    # Generate the set of removable edges
    E2 = generateE2(g, alphaE2)

    # Create IP, write it to file, and solve it with CPLEX
    print(lpname)
    ip, variable_names = createIP(g, E2, lp_dir + "/" + lpname)

    if solveInstance:
        start_time = ip.get_time()
        phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj = solveIP(ip, timelimit, mipgap, relgap_pool, maxsols, threads)
        end_time = ip.get_time()
        total_time = end_time - start_time
 
        num_solutions = ip.solution.pool.get_num()       
        with open(sol_dir + "/" + lpname + ".sol", "w+") as sol_file:
            sol_file.write(("%s,%s,%g,%s,%g,%g,%d,%g\n" % (
                lpname, 
                phase1_status, 
                phase1_gap, 
                phase2_status, 
                phase2_gap, 
                phase2_bestobj, 
                num_solutions, 
                total_time)))

        if num_solutions >= 1:
            # Collect solutions from pool
            solutions_matrix = np.zeros((num_solutions, len(ip.solution.pool.get_values(0))))
            objval_arr = np.zeros(num_solutions)
            for sol_idx in range(num_solutions):
                sol_objval = ip.solution.pool.get_objective_value(sol_idx)
                objval_arr[sol_idx] = sol_objval 
                # if sol_objval > 0:
                solutions_matrix[sol_idx] = ip.solution.pool.get_values(sol_idx)
            solutions_obj_matrix = np.concatenate((np.expand_dims(objval_arr, axis=0).T, solutions_matrix), axis=1)
            # with open(sol_dir + "/" + lpname + ".sol", "ab") as sol_file:
                # np.savetxt(sol_file, solutions_obj_matrix, fmt="%2.f")
            with open(sol_dir + "/" + lpname + ".npz", 'wb') as f:
                np.savez_compressed(f, solutions=solutions_obj_matrix)
            # with open(sol_dir + "/" + lpname + ".npz", 'rb') as f:
            #     a = np.load(f)
            #     print(a["solutions"].shape)

            # Create variable-constraint graph
            vcg = extractVCG(g, E2, ip)

            nx.write_gpickle(vcg, data_dir + "/" + lpname + ".pk")

            vcg2=nx.read_gpickle(data_dir + "/" + lpname + ".pk")
            # print(["(%s, %g)" % (n, d['bias']) for n, d in vcg2.nodes(data=True) if d['bipartite']==0])