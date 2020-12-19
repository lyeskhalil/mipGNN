#!/bin/bash

#to get the right reqs file: pip freeze > requirements.txt

VENVS_DIR="/home/khalile2/projects/def-khalile2/khalile2/venvs_elias/"
VENV_NAME="mipgnn2"

module load python/3.7
virtualenv --no-download $VENVS_DIR/$VENV_NAME
source $VENVS_DIR/$VENV_NAME/bin/activate
pip install --no-index --upgrade pip

pip install numpy --no-index
pip install networkx --no-index

pip install multiprocess
pip install submitit

pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorboard

pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric

python /home/khalile2/software/CPLEX_Studio1210/python/setup.py install


## execution
module load python/3.7
source /home/khalile2/projects/def-khalile2/khalile2/venvs_elias/mipgnn2/bin/activate
