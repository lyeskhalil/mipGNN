import spo_train
import submitit

def add(a, b):
    return a + b

executor = submitit.AutoExecutor(folder="log_test")
print(executor.which())

executor.update_parameters(
	additional_parameters={"account": "rrg-khalile2"}, 
	timeout_min=719,
	mem_per_cpu=16,
	cpus_per_task=10)

job = executor.submit(spo_train.main, 
	[
	'-method', 'spo',
	'-data_train_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/train',
	'-data_validation_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/valid',
	'-output_dir', 'spo_torch_valid_polydeg2',
	'-nn_poly_degree', '1',
	'-nn_depth', '2',
	'-nn_width', '50',
	'-nn_lr_decay', '1',
	'-nn_lr_init', '1e-2',
	'-nn_reg', '0',
	'-nn_batchsize', '10',
	'-nn_poolsize', '10'
	])  
print(job.job_id)  # ID of your job

#output = job.result()  # waits for completion and returns output
