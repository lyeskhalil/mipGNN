#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/projects/def-khalile2/software/geos/bin/lib64
source ~/.bashrc
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_discretenet.txt
pip install --no-index submitit
python slurm_datagen.py
