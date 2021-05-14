
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


from model_execution.predict import get_prediction
from torchmetrics import F1, Precision, Recall, Accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




n = os.listdir("../../data_new/data_graphsonly/gisp/p_hat300-2.clq/train/")[0]
graph = nx.read_gpickle("../../data_new/data_graphsonly/gisp/p_hat300-2.clq/train/" + n)

print(get_prediction("trained_p_hat300-2", graph))

exit()

