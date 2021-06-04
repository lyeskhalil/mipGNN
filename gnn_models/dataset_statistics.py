import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import networkx as nx

name_list = [
    "p_hat300-2.clq_train",
    "gisp_C250.9.clq_train",
    "keller4.clq_train",
    "hamming8-4.clq_train",
    "gen200_p0.9_55.clq_train",
    "gen200_p0.9_44.clq_train",
    "C125.9.clq_train",
    "p_hat300-1.clq_train",
    "brock200_4.clq_train",
    "brock200_2.clq_train",
    "L_n200_p0.02_c500_train"
]

dataset_list = [
    "../data_new/data_graphsonly/gisp/p_hat300-2.clq/train/",
    "../data_new/data_graphsonly/gisp/C250.9.clq/train/",
    "../data_new/data_graphsonly/gisp/keller4.clq/train/",
    "../data_new/data_graphsonly/gisp/hamming8-4.clq/train/",
    "../data_new/data_graphsonly/gisp/gen200_p0.9_55.clq/train/",
    "../data_new/data_graphsonly/gisp/gen200_p0.9_44.clq/train/",
    "../data_new/data_graphsonly/gisp/C125.9.clq/train/",
    "../data_new/data_graphsonly/gisp/p_hat300-1.clq/train/",
    "../data_new/data_graphsonly/gisp/brock200_4.clq/train/",
    "../data_new/data_graphsonly/gisp/brock200_2.clq/train/",
    "../data_new/data_graphsonly/fcmnf/L_n200_p0.02_c500/train/",
]

# Loop over datasets.
for i in range(11):
    name = name_list[i]
    pd = dataset_list[i]

    num_graphs = len(os.listdir(pd))
    num_vars_nodes = 0
    num_cons_nodes = 0
    num_edges = 0

    # Loop over file in datasets.
    for num, filename in enumerate(os.listdir(pd)):

        # print(filename, num, num_graphs)

        # Get graph.
        graph = nx.read_gpickle(pd + filename)

        # Make graph directed.
        graph = nx.convert_node_labels_to_integers(graph)
        graph = graph.to_directed() if not nx.is_directed(graph) else graph

        for i, (node, node_data) in enumerate(graph.nodes(data=True)):

            # Node is a variable node.
            if node_data['bipartite'] == 0:
                num_vars_nodes += 1

            # Node is constraint node.
            elif node_data['bipartite'] == 1:
                num_cons_nodes += 1

        num_edges += graph.number_of_edges()

    print(num_vars_nodes / num_graphs, num_cons_nodes / num_graphs, num_edges / num_edges)
