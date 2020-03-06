import os
import sys
import cplex
from cplex.exceptions import CplexError
import networkx as nx
import random
import time

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
            g.node[node]['revenue'] = random.randint(1,100)
        for u,v,edge in g.edges(data=True):
            edge['cost'] = (g.node[u]['revenue'] + g.node[v]['revenue'])/float(setParam)
    elif whichSet == 'SET2':
        for node in g.nodes():
            g.node[node]['revenue'] = float(setParam)
        for u,v,edge in g.edges(data=True):
            edge['cost'] = 1.0

def generateE2(g, alphaE2):
    E2 = set()
    for edge in g.edges():
        if random.random() <= alphaE2:
            E2.add(edge)
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
        objective_coeffs.append(g.node[node]['revenue'])
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
    return ip

def solveIP(ip, timelimit):
    ip.parameters.timelimit.set(timelimit)

    # disable all cplex output
#     ip.set_log_stream(None)
#     ip.set_error_stream(None)
#     ip.set_warning_stream(None)
#     ip.set_results_stream(None)

    try:
        ip.solve()
    except CplexError as exc:
        print(exc)
        return

    return ip.solution.get_objective_value(), ip.solution.get_status(), ip.solution.MIP.get_mip_relative_gap()

if __name__ == "__main__":
    instance = None
    exp_dir = None
    min_n = None
    max_n = None
    er_prob = 0.1
    whichSet = 'SET2'
    setParam = 100.0
    alphaE2 = 0.75
    timelimit = 7200.0
    solveInstance = False
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
        
    # Seed generator
    random.seed(seed)

    print(whichSet)
    print(setParam)
    print(alphaE2)
    
    if instance is None:
        # Generate random graph
        numnodes = random.randint(min_n, max_n)
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
    ip = createIP(g, E2, lp_dir + "/" + lpname)

    if solveInstance:
        cpx_sol, cpx_status, cpx_gap = solveIP(ip, timelimit)
        with open(sol_dir + "/" + lpname + ".sol", "w+") as sol_file:
            sol_file.write(("%s,%d,%g,%g" % (lpname, cpx_status, cpx_gap, cpx_sol)))
