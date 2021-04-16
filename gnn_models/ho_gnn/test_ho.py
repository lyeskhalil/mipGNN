import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
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

    for i,e in enumerate(graph.edges):
        graph_new.add_node(i, type="AC")

    for i,v in enumerate(graph.nodes):
        graph_new.add_node((i,i), type="VV")