import networkx as nx
import torch_geometric

# pickle file containing the bipartite graph corresponding to a single GISP instance
# the last integer in the filename refers to the random seed that generated this instance
data_path = "DATA/test/C125.9.clq_SET2_0.75_100_0.pk"

# vcg is the Variable-Constraint bipartite graph
vcg = nx.read_gpickle(data_path)

# Each node of vcg has a bipartite attribute = 0 for "variable nodes", 1 for "constraint nodes"
# Each node/edge of vcg is identified by its name (a string)
# Variable nodes are named by the corresponding variable's name in the actual MIP instance; they have the following attributes, in addition to bipartite=0: 
#  'bias', the label we want predict, continuous in [0,1]
#  'objcoeff', the objective function coefficient in the MIP, assuming minimization
# Constraint nodes are named cx, where x is an integer denoting the index of the constraint in the actual MIP instance; they have the following attributes, in addition to bipartite=1: 
#  'rhs', the right-hand side value of the constraint, assuming <= constraints
# Edges of vcg have the following attributes:
#  'coeff', the coefficient of the variable in the constraint matrix of Ax <= b 

# Example: accessing the biases of the variable nodes
print([node_data['bias'] for node, node_data in vcg.nodes(data=True) if node_data['bipartite']==0])

# Converting the networkx graph into a torch geometric data object; untested
# https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx
vcg_torch = torch_geometric.data.from_networkx(vcg)