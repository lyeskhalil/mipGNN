import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import networkx as nx

data_path = "../../DATA1/er_SET2/200_200/alpha_0.75_setParam_100/train/"

for num, filename in enumerate(os.listdir(data_path)):
    print(num)
    # Get graph.
    graph = nx.read_gpickle(data_path + filename)

    # Make graph directed.
    graph = nx.convert_node_labels_to_integers(graph)
    graph = graph.to_directed() if not nx.is_directed(graph) else graph

    graph_new = nx.Graph()

    matrices_vv_cv = []

    for i, (u, v) in enumerate(graph.edges):
        if graph.nodes[v]['bipartite'] == 0:
            graph_new.add_node(i, type="VC", first=u, second=v)
        else:
            graph_new.add_node(i, type="CV", first=u, second=v)

    for i, v in enumerate(graph.nodes):
        if graph.nodes[v]['bipartite'] == 0:
            graph_new.add_node(i, type="VV", first=v, second=v)
        else:
            graph_new.add_node(i, type="CC", first=v, second=v)


    for i, v, data in enumerate(graph_new.nodes(data=True)):
        print(data)


