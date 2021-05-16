import inference
import submitit
import glob
import os
from cplex.exceptions import CplexError
from pathlib import Path
import time
import math


def combine_jobs(dict_list):
    for counter, dict_cur in enumerate(dict_list):
        print("job %d/%d" % (counter+1, len(dict_list)))
        #inference.mipeval(**dict_cur)
        try:
            inference.mipeval(**dict_cur)
        except CplexError as exc:
            print("errjob %d/%d" % (counter+1, len(mps_paths_subset)))
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

problem_class = "gisp" #"fcmnf/L_n200_p0.02_c500" #"gisp"
data_main_path = "../datagen/data/"
data_specific_path = "%s/p_hat300-2.clq/mipeval/" % (problem_class.replace('/','_'))
path_prefix = "%s/%s" % (data_main_path, data_specific_path)
model_path = "../gnn_models/EdgeConv/trained_p_hat300-2"
# mps_paths = [str(path) for path in Path(path_prefix).rglob('*.mps')]
output_dir = "OUTPUT_new/"

barebones = 0
configs = {}
configs['default-0'] = {'method':'default', 'barebones':barebones}
configs['node_selection-0-100'] = {'method':'node_selection', 'barebones':barebones, 'freq_best':100}
configs['node_selection-0-10'] = {'method':'node_selection', 'barebones':barebones, 'freq_best':10}
configs['primal_mipstart-0'] = {'method':'primal_mipstart', 'barebones':barebones}
configs['branching_priorities-0'] = {'method':'branching_priorities', 'barebones':barebones}
configs['default_emptycb-0'] = {'method':'default_emptycb', 'barebones':barebones}

dict_list = []
for mps_path in Path(path_prefix).glob('*.mps'):
    instance_path_noext = os.path.splitext(mps_path)[0]
    vcg_path = "%s_graph_bias.pkl" % instance_path_noext
    for config_name, config_dict in configs.items():
        config_dict_final = dict(config_dict)
        config_dict_final['timelimit'] = timelimit
        config_dict_final['memlimit'] = memlimit

        if 'default' not in config_dict['method']:
            if not Path(vcg_path).is_file():
                print('Failed for %s' % mps_path)
                print(config_dict)
                continue
            config_dict_final['model'] = model_path
            config_dict_final['graph'] = vcg_path

        # instance, logfile
        instance_name_noext = os.path.splitext(os.path.basename(mps_path))[0]
        config_dict_final['instance'] = str(mps_path)
        config_dict_final['logfile'] = '%s/%s/%s/' % (output_dir, data_specific_path, config_name)#, instance_name_noext)
        os.makedirs(config_dict_final['logfile'], exist_ok=True)
        config_dict_final['logfile'] = '%s/%s.out' % (config_dict_final['logfile'], instance_name_noext)
        #config_dict_final['logfile'] = 'sys.stdout'
        dict_list += [config_dict_final]

#combine_jobs([dict_list[1]])
#exit()

num_tasks = len(dict_list)
chunk_size = math.ceil(num_tasks / 1000.0)
print("Chunks being mapped. chunk_size = %d" % chunk_size)
dict_listoflistsofdicts = list(chunks(dict_list, chunk_size))

timeout_min=math.ceil(math.ceil(timelimit/60.0)*chunk_size*2)
print("timeout_min = %d" % timeout_min)

#print(dict_listoflistsofdicts)
#exit()

print("Submitit initialization...")
executor = submitit.AutoExecutor(folder="slurm_logs_mipeval")
print(executor.which())

executor.update_parameters(
        array_parallelism=1000,
        additional_parameters={"account": "rrg-khalile2"},
        timeout_min=timeout_min,
        mem_gb=mem_gb,
        cpus_per_task=num_cpus)

print("Submitit submitting job array...")
jobs = executor.map_array(combine_jobs, dict_listoflistsofdicts)
