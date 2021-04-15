

import warnings, sys
import argparse
import os.path as osp
from ogb.graphproppred import PygGraphPropPredDataset
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from torch_geometric.nn import WLConv
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
warnings.filterwarnings('ignore', category=ConvergenceWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
torch.manual_seed(42)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='ENZYMES')
data = Batch.from_data_list(dataset)
class WL(torch.nn.Module):
    def __init__(self, num_layers):
        super(WL, self).__init__()
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])
    def forward(self, x, edge_index, batch=None):
        x = torch.ones((x.size(0),1))
        hists = []
        colors = []
        for conv in self.convs:
            x = conv(x, edge_index)
            colors.append(x)
            hists.append(conv.histogram(x, batch, norm=True))
        return colors, hists
wl = WL(num_layers=1)
colors, hists = wl(data.x, data.edge_index, data.batch)
print(colors[0].max().item())
colors = []
for data in dataset:
    c, h = wl(data.x, data.edge_index)
    colors.append(c[0].max().item())
print(max(colors))
print ("#################")
dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = "ogbg-molhiv")
data = Batch.from_data_list(dataset)
wl = WL(num_layers=1)
colors, hists = wl(data.x, data.edge_index, data.batch)
print(colors[0].max().item())
colors = []
for data in dataset:
    c, h = wl(data.x, data.edge_index)
    colors.append(c[0].max().item())
print(max(colors))
