import bias_search
import submitit
import glob
import os


def combine_jobs(mps_paths_subset, timelimit, threads, memlimit):
    for mps_path in mps_paths_subset:
        bias_search.search(mps_path, timelimit[0], threads[0], memlimit[0])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


problem_class = "gisp"

graphs_path = '/home/khalile2/projects/def-khalile2/software/DiscreteNet/discretenet/problems/gisp/graphs'
graphs_filenames = [os.path.basename(graph_fullpath) for graph_fullpath in glob.glob(graphs_path + "/*.clq")] 
print(graphs_filenames)

print("Fetching mps_paths...")
mps_paths = []
for graph in graphs_filenames:
    for data_type in ['test']:
        #random_seed = int(data_type == 'test')
        path_prefix = "data/%s/%s/%s/" % (problem_class, graph, data_type)
        print(path_prefix)
        for instance in glob.glob(path_prefix + "/*.mps"):
            mps_paths += [instance]

mem_gb=8
timelimit = [1800]*len(mps_paths)
threads = [4]*len(mps_paths)
memlimit = [int(mem_gb/2.0)*1024]*len(mps_paths)

#jobs = executor.map_array(bias_search.search, mps_paths, timelimit, threads, memlimit)

print("Chunks being mapped...")
chunk_size = 13
mps_paths_subsets, timelimit_subsets, threads_subsets, memlimit_subsets = list(chunks(mps_paths, chunk_size)), list(chunks(timelimit, chunk_size)), list(chunks(threads, chunk_size)), list(chunks(memlimit, chunk_size))

timeout_min=70*chunk_size
num_cpus = threads[0]

print("Submitit initialization...")
executor = submitit.AutoExecutor(folder="slurm_logs_bias_chunks2_test")
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

