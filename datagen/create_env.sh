#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/projects/def-khalile2/software/geos/bin/lib64
source ~/.bashrc
module load python/3.6
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_discretenet.txt
pip install submitit
python /home/khalile2/software/CPLEX_Studio1210/python/setup.py install
#python slurm_datagen.py
