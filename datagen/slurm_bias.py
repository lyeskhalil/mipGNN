import bias_search
import submitit
import glob
import os
from cplex.exceptions import CplexError
from pathlib import Path
import time


def combine_jobs(mps_paths_subset, timelimit, threads, memlimit):
    for counter, mps_path in enumerate(mps_paths_subset):
        print("job %d/%d" % (counter, len(mps_paths_subset)))
        try:
            bias_search.search(mps_path, timelimit[0], threads[0], memlimit[0])
        except CplexError as exc:
            print("errjob %d/%d" % (counter, len(mps_paths_subset)))
            print(exc)
            continue
        except Exception as e:
            print("unexpected error")
            continue


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


problem_class = "gisp" #"fcmnf/L_n200_p0.02_c500" #"gisp"
path_prefix = "data/%s/" % (problem_class.replace('/','_'))
mps_paths = [str(path) for path in Path(path_prefix).rglob('*.mps')]

if False:
	paths_notupdated = []
	paths_notexist = []
	for mps_path in Path(path_prefix).rglob('*.mps'):
		instance_name_noext = os.path.splitext(mps_path)[0]
		vcg_labeled_path = "%s_graph_bias.pkl" % instance_name_noext
		if not Path(vcg_labeled_path).is_file():
			paths_notexist += [str(mps_path)]
			continue
		timediff = (time.time() - os.path.getmtime(vcg_labeled_path))/3600.0
		if timediff > 20:
			paths_notupdated += [str(mps_path)]

	print(len(paths_notupdated), len(paths_notexist))
	mps_paths = paths_notupdated + paths_notexist


mem_gb=8
timelimit = [1800]*len(mps_paths)
threads = [4]*len(mps_paths)
memlimit = [int(mem_gb/2.0)*1024]*len(mps_paths)

#jobs = executor.map_array(bias_search.search, mps_paths, timelimit, threads, memlimit)

print("Chunks being mapped...")
chunk_size = 1
mps_paths_subsets, timelimit_subsets, threads_subsets, memlimit_subsets = list(chunks(mps_paths, chunk_size)), list(chunks(timelimit, chunk_size)), list(chunks(threads, chunk_size)), list(chunks(memlimit, chunk_size))

timeout_min=90#70*chunk_size
num_cpus = threads[0]

print("Submitit initialization...")
executor = submitit.AutoExecutor(folder="slurm_logs_bias_%s" % (problem_class))
print(executor.which())

executor.update_parameters(
        array_parallelism=1000,
        additional_parameters={"account": "rrg-khalile2"},
        timeout_min=timeout_min,
        mem_gb=mem_gb,
        cpus_per_task=num_cpus)

print("Submitit submitting job array...")
jobs = executor.map_array(combine_jobs, mps_paths_subsets, timelimit_subsets, threads_subsets, memlimit_subsets)
exit()

