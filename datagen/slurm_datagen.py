import gen
import submitit
import glob
import os

timeout_min=60
mem_gb=8

problem_class = "fcmnf"
num_cpus = 16
n_instances = 1000
n_jobs = num_cpus

executor = submitit.AutoExecutor(folder="slurm_logs_datagen_%s" % (problem_class))
print(executor.which())

executor.update_parameters(
        additional_parameters={"account": "rrg-khalile2"}, 
        timeout_min=timeout_min,
        mem_gb=mem_gb,
        cpus_per_task=num_cpus)

if problem_class == 'gisp':
    graphs_path = '/home/khalile2/projects/def-khalile2/software/DiscreteNet/discretenet/problems/gisp/graphs'
    graphs_filenames = [os.path.basename(graph_fullpath) for graph_fullpath in glob.glob(graphs_path + "/*.clq")] 
    print(graphs_filenames)

    for graph in graphs_filenames:
        for data_type in ['train', 'test']:
            random_seed = int(data_type == 'test')
            path_prefix = "data/%s/%s/%s/" % (problem_class, graph, data_type)
            print(path_prefix)
            # generate(random_seed, path_prefix, graph_instance, n_instances, n_jobs)
            job = executor.submit(gen.generate, problem_class, random_seed, path_prefix, graph, n_instances, n_jobs)
            print(job.job_id)
elif problem_class == 'fcmnf':
    for data_type in ['train', 'test']:
        random_seed = int(data_type == 'test')
        path_prefix = "data/%s/%s/%s/" % (problem_class, "L_n200_p0.02_c500", data_type)
        print(path_prefix)
        job = executor.submit(gen.generate, problem_class, random_seed, path_prefix, "", n_instances, n_jobs)
        print(job.job_id)
