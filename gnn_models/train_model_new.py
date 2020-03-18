import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx


import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

torch.autograd.set_detect_anomaly(True)

from gnn_models.mip_architecture import Net

class GISR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ER1"

    @property
    def processed_file_names(self):
        return "ER1"

    def download(self):
        pass

    def process(self):
        data_list = []

        path = '../gisp_generator/DATA/er_200/'

        total = len(os.listdir(path))

        for num, filename in enumerate(os.listdir(path)):
            print(filename, num, total)

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
            rhss = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):

                # Node is a variable.
                if node_data['bipartite'] == 0:
                    y.append(node_data['bias'])
                    node_type.append(0)
                    assoc_var.append(i)
                    coeff = node_data['objcoeff']

                    # TODO: Scaling meaingful?
                    var_feat.append([coeff / 100.0, graph.degree[i]])

                # Node is constraint.
                else:
                    node_type.append(1)
                    assoc_con.append(i)
                    rhs = node_data['rhs']
                    rhss.append(rhs)
                    con_feat.append([rhs, graph.degree[i]])

            y = torch.from_numpy(np.array(y)).to(torch.float).to(torch.float)
            data.y = y
            data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
            data.node_types = torch.from_numpy(np.array(node_type)).to(torch.long)
            data.assoc_var = torch.from_numpy(np.array(assoc_var)).to(torch.long)
            data.assoc_con = torch.from_numpy(np.array(assoc_con)).to(torch.long)
            data.rhs = torch.from_numpy(np.array(rhss)).to(torch.float)

            edge_types = []
            edge_features = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):

                if graph.nodes[s]['bipartite']:
                    edge_types.append(0)
                    edge_features.append([edge_data['coeff']])
                else:
                    edge_types.append(1)
                    edge_features.append([edge_data['coeff']])

            data.edge_types = torch.from_numpy(np.array(edge_types)).to(torch.long)
            data.edge_features = torch.from_numpy(np.array(edge_features)).to(torch.float)

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


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DS')
dataset = GISR(path, transform=MyTransform()).shuffle()
dataset.data.y = torch.log(dataset.data.y + 1.0)
print(len(dataset))


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DS')
dataset = GISR(path, transform=MyTransform()).shuffle()
dataset.data.y = torch.log(dataset.data.y + 1.0)
print(len(dataset))

train_dataset = dataset[0:800].shuffle()
val_dataset = dataset[800:900].shuffle()
test_dataset = dataset[900:].shuffle()

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)


print("### DATA LOADED.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

print("### SETUP DONE.")

def train():
    model.train()
    total_loss = 0
    total_loss_mae = 0
    loss = torch.nn.MSELoss()
    mse = RMSELoss()
    #mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    #mse = torch.nn.SmoothL1Loss()


    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)

        loss = mse(out, data.y)
        loss.backward()

        total_loss += loss.item() * data.num_graphs
        total_loss_mae += mae(out, data.y).item() * data.num_graphs

        optimizer.step()

    return total_loss_mae / len(train_loader.dataset), total_loss / len(train_loader.dataset)



def test(loader):
    model.eval()
    error = 0
    l1 = torch.nn.L1Loss()

    for data in loader:
        data = data.to(device)
        out = model(data)

        loss = l1(torch.exp(out) - 1.0, torch.exp(data.y) - 1.0)
        error += loss.item() * data.num_graphs

    return error / len(loader.dataset)


best_val_error = None

test_error = test(test_loader)
print(test_error)

for epoch in range(1, 500):
    lr = scheduler.optimizer.param_groups[0]['lr']
    mae, loss = train()

    if epoch == 12:
        for param_group in optimizer.param_groups:
           param_group['lr'] = 0.5 * param_group['lr']

    val_error = test(val_loader)

    if best_val_error is None or val_error < best_val_error:
        test_error = test(test_loader)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, loss, mae, test_error))

# torch.save(model.state_dict(), "train_mip")


