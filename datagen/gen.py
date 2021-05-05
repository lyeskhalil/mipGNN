import sys
sys.path.extend(["/home/khalile2/projects/def-khalile2/software/DiscreteNet"])
from discretenet.problems.gisp import GISPGenerator
from discretenet.problems.fcmnf import FCMNFGenerator
import argparse
import networkx as nx

def generate(problem_class, random_seed, path_prefix, graph_instance, n_instances, n_jobs):
    if problem_class == 'gisp':
        generator = GISPGenerator(
                random_seed=random_seed,
                path_prefix=path_prefix,
                which_set="SET2",
                graph_instance=graph_instance,
                set_param=100.0,
                alpha=0.75
                )

    elif problem_class == 'fcmnf':
        generator = FCMNFGenerator(
            random_seed=random_seed,
            path_prefix=path_prefix,
            min_n=200,
            max_n=200,
            er_prob=0.02,
            variable_costs_range_lower=11,
            variable_costs_range_upper=50,
            commodities_quantities_range_lower=10,
            commodities_quantities_range_upper=100,
            fixed_to_variable_ratio=1000,
            edge_upper=500, #Loose=500, Tight=5
            num_commodities=100
            )

    else:
        print("PROBLEM UNDEFINED, ABORT")

    generator(
            n_instances=n_instances, 
            n_jobs=n_jobs, 
            save=True, 
            save_params=True, 
            save_features=False, 
            return_instances=False
            )

    # for instance in instances:
    #     print(instance.get_name())
    #     vcg = instance.get_variable_constraint_graph()
    #     print(vcg.number_of_nodes())
    #     nx.write_gpickle(vcg, "%s/%s_graph.pkl" % (path_prefix, instance.get_name())

generate("fcmnf", 0, "data/debug/fcmnf/", "", 1, 1)
