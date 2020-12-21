import spo_train
import spo_utils
import submitit
import random
from random import sample 

output_dir = 'spo_torch_polydeg2_kernel_hypersearch'
num_cpus = 10

executor = submitit.AutoExecutor(folder="log_%s" % output_dir)
print(executor.which())

executor.update_parameters(
	additional_parameters={"account": "rrg-khalile2"}, 
	timeout_min=719,
	mem_gb=16,
	cpus_per_task=num_cpus)

#dict_allvals = {'-nn_depth': ['1','2','3'], '-nn_width': ['10','20','40', '80'], '-nn_lr_decay': ['0', '1'], '-nn_lr_init': ['1e-3', '5e-3'], '-nn_reg': ['0', '1'], '-nn_batchsize': ['5', '10', '20', '50', '100'], '-nn_sgd_nesterov': ['0', '1'], '-nn_sgd_momentum': ['0', '0.2', '0.4', '0.8']}

dict_allvals = {'-nn_poly_degree': ['2'], '-nn_depth': ['0'], '-nn_width': ['0'], '-nn_lr_decay': ['0', '1'], '-nn_lr_init': ['1e-3', '5e-3', '1e-2', '1e-1', '1e0'], '-nn_reg': ['1e-6', '1e-4', '1e-2', '1e-1', '0', '1e0'], '-nn_batchsize': ['10'], '-nn_sgd_nesterov': ['1'], '-nn_sgd_momentum': ['0.2', '0.4', '0.8']}

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

random.seed(0)
configs = sample(configs, min([500, len(configs)])) 

for config in configs:
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

	job = executor.submit(spo_train.main, arg_list)
 
	print(job.job_id)  # ID of your job

#output = job.result()  # waits for completion and returns output
