import sys
sys.path.extend(["/home/khalile2/projects/def-khalile2/software/DiscreteNet"])
from discretenet.problems.gisp import GISPGenerator
from discretenet.problems.fcmnf import FCMNFGenerator
import argparse
import networkx as nx

def generate(random_seed, path_prefix, graph_instance, n_instances, n_jobs):
    generator = GISPGenerator(
            random_seed=random_seed,
            path_prefix=path_prefix,
            which_set="SET2",
            graph_instance=graph_instance,
            set_param=100.0,
            alpha=0.75
            )

    instances = generator(
            n_instances=n_instances, 
            n_jobs=n_jobs, 
            save=True, 
            save_params=True, 
            save_features=False, 
            return_instances=True
            )

    for instance in instances:
        print(instance.get_name())
        vcg = instance.get_variable_constraint_graph()
        print(vcg.number_of_nodes())
        nx.write_gpickle(vcg, "%s/%s_graph.pkl" % (path_prefix, instance.get_name()))
