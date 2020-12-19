import spo_train
import submitit

def add(a, b):
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")
print(executor.which())
# set timeout in min, and partition for running the job
executor.update_parameters(additional_parameters={"account": "rrg-khalile2"}, timeout_min=719)
job = executor.submit(spo_train.main, [])  # will compute add(5, 7)
print(job.job_id)  # ID of your job

#output = job.result()  # waits for completion and returns output
#assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
