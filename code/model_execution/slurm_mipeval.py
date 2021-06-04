import inference
import glob
import os
from cplex.exceptions import CplexError
from pathlib import Path
import time
import math
from concurrent import futures


def combine_jobs(dict_list):
    for counter, dict_cur in enumerate(dict_list):
        print("job %d/%d" % (counter+1, len(dict_list)))
        try:
            inference.mipeval(**dict_cur)
        except CplexError as exc:
            print("errjob %d/%d" % (counter+1, len(dict_list)))
            print(exc)
            continue
        except Exception as e:
            print("unexpected error")
            continue


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


num_cpus = 1
mem_gb = 8
timelimit = 1800
memlimit = int(mem_gb/2.0)*1024

problem_class = "gisp"
data_main_path = "../data/"
output_dir = "OUTPUT/"

model_name = "SG"
models_path = "../gnn_models/models_pretrained/%s" % (model_name)
model_hyperparams = "train0.0"

barebones = 0
configs = {}
configs['default-%d' % (barebones)] = {'method':['default'], 'barebones':barebones}
configs['node_selection-%d-100' % (barebones)] = {'method':['node_selection'], 'barebones':barebones, 'freq_best':100}
configs['primal_mipstart-%d' % (barebones)] = {'method':['primal_mipstart'], 'barebones':barebones, 'mipstart_strategy':'repair'}
configs['branching_priorities-%d' % (barebones)] = {'method':['branching_priorities'], 'barebones':barebones}

dict_list = []

if problem_class == "gisp":
    graphs_filenames = [os.path.basename(dirpath) for dirpath in glob.glob('../data/gisp/*')]
elif problem_class == "fcmnf":
    graphs_filenames = ["L_n200_p0.02_c500"]

for graph in graphs_filenames:
    data_specific_path = "%s/%s/mipeval/" % (problem_class.replace('/','_'), graph)
    path_prefix = "%s/%s" % (data_main_path, data_specific_path)

    model_prefix = models_path if 'C250.9.clq' not in graph else models_path + '_gisp'
    
    graph_trained = graph

    for mps_path in Path(path_prefix).glob('*.mps'):
        instance_path_noext = os.path.splitext(mps_path)[0]
        #vcg_path = "%s_graph_bias.pkl" % instance_path_noext
        instance_params_path = "%s_parameters.pkl" % instance_path_noext
        for config_name, config_dict in configs.items():
            config_dict_final = dict(config_dict)
            config_dict_final['timelimit'] = timelimit
            config_dict_final['memlimit'] = memlimit

            if 'default' not in config_dict['method']:
                model_path = '%s_%s_%s' % (model_prefix, graph_trained, model_hyperparams)
                if not Path(instance_params_path).is_file() or not Path(model_path).is_file():
                    print('Failed for %s' % mps_path)
                    print(model_path)
                    print(config_dict)
                    continue
                config_dict_final['model'] = model_path
                config_dict_final['instance_params'] = instance_params_path
                #config_dict_final['graph'] = vcg_path

            # instance, logfile
            instance_name_noext = os.path.splitext(os.path.basename(mps_path))[0]
            config_dict_final['instance'] = str(mps_path)
            config_dict_final['logfile'] = '%s/%s/%s/' % (output_dir, data_specific_path, config_name)#, instance_name_noext)
            os.makedirs(config_dict_final['logfile'], exist_ok=True)
            config_dict_final['logfile'] = '%s/%s.out' % (config_dict_final['logfile'], instance_name_noext)
            dict_list += [config_dict_final]

print("Submitting num_workers parallel jobs...")
num_workers = 1
executor = futures.ThreadPoolExecutor(max_workers=num_workers)
jobs = executor.map_array(combine_jobs, dict_list)
