import spo_train
import spo_utils
import submitit
from random import sample 

executor = submitit.AutoExecutor(folder="log_test")
print(executor.which())

executor.update_parameters(
	additional_parameters={"account": "rrg-khalile2"}, 
	timeout_min=719,
	mem_gb=16,
	cpus_per_task=10)

dict_allvals = {'-nn_depth': ['1','2','3'], '-nn_width': ['10','20','40', '80'], '-nn_lr_decay': ['0', '1'], '-nn_lr_init': ['1e-3', '5e-3'], '-nn_reg': ['0', '1'], '-nn_batchsize': ['5', '10', '20', '50', '100'], '-nn_sgd_nesterov': ['0', '1'], '-nn_sgd_momentum': ['0', '0.2', '0.4', '0.8']}
configs = list(spo_utils.dict_product(dict_allvals))

print("total number of configurations =", len(configs))
# job = executor.submit(spo_train.main, 
# 	[
# 	'-method', 'spo',
# 	'-data_train_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/train',
# 	'-data_validation_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/valid',
# 	'-output_dir', 'spo_torch_valid_polydeg2',
# 	'-nn_poly_degree', '1',
# 	'-nn_depth', '2',
# 	'-nn_width', '50',
# 	'-nn_lr_decay', '0',
# 	'-nn_lr_init', '1e-2',
# 	'-nn_reg', '0',
# 	'-nn_batchsize', '10',
# 	'-nn_poolsize', '10'
# 	])  

  
configs = sample(configs, 500) 

for config in configs:
	arg_list =
		[
		'-method', 'spo',
		'-data_train_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/train',
		'-data_validation_dir', '../gisp_generator/SPO_DATA/spo_gisp_er/150_150/alpha_0.75_numFeat_10_biasnodes_100_biasedges_10_halfwidth_0.5_polydeg_2/valid',
		'-output_dir', 'spo_torch_polydeg2_hypersearch',
		'-nn_poly_degree', '1',
		'-nn_poolsize', '50'
		]

	for arg, value in config.items():
		arg_list += [arg, value]

	print(arg_list)

	job = executor.submit(spo_train.main, arg_list)
 
	print(job.job_id)  # ID of your job

#output = job.result()  # waits for completion and returns output
