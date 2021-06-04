#!/bin/bash

# Tested with python 3.7, requires virtualenv
# pip install virtualenv

VENVS_DIR="env_dir/"
VENV_NAME="mipgnn"

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

pip install torchmetrics

CPLEX_DIR="path/to/cplex/"
python ${CPLEX_DIR}/CPLEX_Studio1210/python/setup.py install

## To activate the environment:
source $VENVS_DIR/$VENV_NAME/bin/activate
