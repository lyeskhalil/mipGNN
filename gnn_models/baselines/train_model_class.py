import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import torch
import numpy as np
import networkx as nx

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from gnn_models.baselines.mpnn_architecture_class import Net

class GISDS(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISDS, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "EeffRSrr"

    @property
    def processed_file_names(self):
        return "EerffrrRS"

    def download(self):
        pass

    def process(self):
        data_list = []

        path = '../../gisp_generator/DATA/er_200_SET1/'

        total = len(os.listdir(path))

        for num, filename in enumerate(os.listdir(path)):
            print(filename)

            graph = nx.read_gpickle(path + filename)

            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            data = Data(edge_index=edge_index)

            y = []
            node_type = []

            assoc_var = []
            assoc_con = []

            var_feat = []
            con_feat = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):

                # Node is a variable.
                if node_data['bipartite'] == 0:
                    node_type.append(0)
                    assoc_var.append(i)
                    coeff = node_data['objcoeff']

                    if (node_data['bias'] < 0.05):
                        y.append(0)
                    else:
                        y.append(1)

                    # TODO: Maybe scale this
                    var_feat.append([coeff/100.0, graph.degree[i]])
                # Node is constraint.
                else:
                    node_type.append(1)
                    assoc_con.append(i)
                    rhs = node_data['rhs']
                    con_feat.append([rhs, graph.degree[i]])

            data.y = torch.from_numpy(np.array(y)).to(torch.long)
            data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
            data.node_types = torch.from_numpy(np.array(node_type)).to(torch.long)
            data.assoc_var = torch.from_numpy(np.array(assoc_var)).to(torch.long)
            data.assoc_con = torch.from_numpy(np.array(assoc_con)).to(torch.long)

            edge_types = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):

                if graph.nodes[s]['bipartite']:
                    edge_types.append([0, edge_data['coeff']])
                else:
                    edge_types.append([1, edge_data['coeff']])

            data.edge_types = torch.from_numpy(np.array(edge_types)).to(torch.float)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        return self.num_nodes if key in [
            'edge_index', 'assoc_var', 'assoc_con'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        new_data.num_nodes = data.node_types.size(0)
        return new_data




# Prepare data.
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'DS')
dataset = GISDS(path, transform=MyTransform()).shuffle()
len(dataset)

train_dataset = dataset[0:800].shuffle()
val_dataset = dataset[800:900].shuffle()
test_dataset = dataset[900:1000].shuffle()

print(len(val_dataset))

print(1 - test_dataset.data.y.sum().item() / test_dataset.data.y.size(-1))

batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("### DATA LOADED.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=3, min_lr=0.00001)
print("### SETUP DONE.")


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    l = 0

    rec = 0.0
    pre = 0.0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).float().mean().item()
        # rec += metrics.recall_score(data.y.tolist(), pred.tolist())
        # pre += metrics.precision_score(data.y.tolist(), pred.tolist())
        l += 1

    # print(rec/l, pre/l)
    return correct / l


best_val = 0.0
test_acc = 0.0
for epoch in range(1, 501):
    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

    train_loss = train(epoch)
    train_acc = test(train_loader)

    val_acc = test(val_loader)
    if val_acc > best_val:
        best_val = val_acc
        test_acc = test(test_loader)

    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))


