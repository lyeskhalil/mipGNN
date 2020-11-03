#!/bin/bash
#SBATCH --array=1-200
#SBATCH --time=00:05:00
#SBATCH --account=rrg-khalile2
##SBATCH --output=LOGS_SLURM/%x-%j.out
#SBATCH --error=LOGS_SLURM/%x-%j.out
#SBATCH --mem-per-cpu=500M
#SBATCH --cpus-per-task=1

module load python/3.7
source /home/khalile2/projects/def-khalile2/khalile2/venvs_elias/mipgnn2/bin/activate

#SLURM_ARRAY_TASK_ID="1"
#eval "$1"
python generateIP.py -exp_dir er_SET2/200_200/alpha_0.75_setParam_100/test -min_n 200 -max_n 200 -whichSet SET2 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET2/200_200/alpha_0.25_setParam_100/test -min_n 200 -max_n 200 -whichSet SET2 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET2/200_200/alpha_0.5_setParam_100/test -min_n 200 -max_n 200 -whichSet SET2 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET2/300_300/alpha_0.75_setParam_100/test -min_n 300 -max_n 300 -whichSet SET2 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET2/300_300/alpha_0.25_setParam_100/test -min_n 300 -max_n 300 -whichSet SET2 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET2/300_300/alpha_0.5_setParam_100/test -min_n 300 -max_n 300 -whichSet SET2 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET1/400_400/alpha_0.75_setParam_100/test -min_n 400 -max_n 400 -whichSet SET1 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET1/400_400/alpha_0.25_setParam_100/test -min_n 400 -max_n 400 -whichSet SET1 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
python generateIP.py -exp_dir er_SET1/400_400/alpha_0.5_setParam_100/test -min_n 400 -max_n 400 -whichSet SET1 -solve 0 -seed $(($SLURM_ARRAY_TASK_ID+10000)) > /dev/null
