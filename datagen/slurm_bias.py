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
    for data_type in ['train']:
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
chunk_size = 10
mps_paths_subsets, timelimit_subsets, threads_subsets, memlimit_subsets = list(chunks(mps_paths, chunk_size)), list(chunks(timelimit, chunk_size)), list(chunks(threads, chunk_size)), list(chunks(memlimit, chunk_size))

timeout_min=70*chunk_size
num_cpus = threads[0]

print("Submitit initialization...")
executor = submitit.AutoExecutor(folder="slurm_logs_bias_chunks2")
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

#dict_allvals = {'-nn_depth': ['1','2','3'], '-nn_width': ['10','20','40', '80'], '-nn_lr_decay': ['0', '1'], '-nn_lr_init': ['1e-3', '5e-3'], '-nn_reg': ['0', '1'], '-nn_batchsize': ['5', '10', '20', '50', '100'], '-nn_sgd_nesterov': ['0', '1'], '-nn_sgd_momentum': ['0', '0.2', '0.4', '0.8']}

dict_allvals = {'-nn_warmstart_dir': ['SPO_MODELS/2stage/'], '-nn_warmstart_prefix': ['linear'], '-nn_poly_degree': ['1'], '-nn_depth': ['0'], '-nn_width': ['0'], '-nn_lr_decay': ['0', '1'], '-nn_lr_init': ['1e-3', '5e-3', '1e-2', '1e-1', '1e0'], '-nn_reg': ['1e-6', '1e-4', '1e-2', '0'], '-nn_batchsize': ['10'], '-nn_sgd_nesterov': ['1'], '-nn_sgd_momentum': ['0.2', '0.4', '0.8']}

configs = list(spo_utils.dict_product(dict_allvals))

print("total number of configurations =", len(configs))
# job = executor.submit(spo_train.main, 
#       [
#       '-method', 'spo',
#       '-data_train_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/train',
#       '-data_validation_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/valid',
#       '-output_dir', 'spo_torch_valid_polydeg2',
#       '-nn_poly_degree', '1',
#       '-nn_depth', '2',
#       '-nn_width', '50',
#       '-nn_lr_decay', '0',
#       '-nn_lr_init', '1e-2',
#       '-nn_reg', '0',
#       '-nn_batchsize', '10',
#       '-nn_poolsize', '10'
#       ])  

random.seed(0)
configs = sample(configs, min([500, len(configs)])) 

for idx, config in enumerate(configs):
        #if idx < 87:
        #    continue

        print("config", idx)

        arg_list = [
        '-method', 'spo',
        '-data_train_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/train',
        '-data_validation_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/valid',
        '-output_dir', output_dir,
        '-nn_poolsize', str(num_cpus)
        ]

        for arg, value in config.items():
                arg_list += [arg, value]

        print(arg_list)

        while True:
            try:
                job = executor.submit(spo_train.main, arg_list)
                break
            except submitit.core.utils.FailedJobError:
                continue
        print(job.job_id)  # ID of your job

#output = job.result()  # waits for completion and returns output
