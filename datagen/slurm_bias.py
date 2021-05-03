import bias_search
import submitit
import glob
import os

timeout_min=130
mem_gb=4

problem_class = "gisp"
num_cpus = 4

executor = submitit.AutoExecutor(folder="slurm_logs_bias")
print(executor.which())

executor.update_parameters(
        array_parallelism=1000,
        additional_parameters={"account": "rrg-khalile2"}, 
        timeout_min=timeout_min,
        mem_gb=mem_gb,
        cpus_per_task=num_cpus)

graphs_path = '/home/khalile2/projects/def-khalile2/software/DiscreteNet/discretenet/problems/gisp/graphs'
graphs_filenames = [os.path.basename(graph_fullpath) for graph_fullpath in glob.glob(graphs_path + "/*.clq")] 
print(graphs_filenames)

mps_paths = []
for graph in graphs_filenames:
    for data_type in ['train', 'test']:
        #random_seed = int(data_type == 'test')
        path_prefix = "data/%s/%s/%s/" % (problem_class, graph, data_type)
        print(path_prefix)
        for instance in glob.glob(path_prefix + "/*.mps"):
            # generate(random_seed, path_prefix, graph_instance, n_instances, n_jobs)
            #job = executor.submit(gen.generate, random_seed, path_prefix, graph, n_instances, n_jobs)
            mps_paths += [instance]
        break
    break
timelimit = [7200]*len(mps_paths)
threads = [4]*len(mps_paths)
memlimit = [mem_gb*1000]*len(mps_paths)

jobs = executor.map_array(bias_search.search, mps_paths, timelimit, threads, memlimit)
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
