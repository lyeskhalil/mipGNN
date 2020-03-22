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
from gnn_models.mip_alternating_arch import Net


# Dataset and preprocessing.
class GISR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "TESdT411"

    @property
    def processed_file_names(self):
        return "TESdT422"

    def download(self):
        pass

    def process(self):
        data_list = []

        path = '../gisp_generator/DATA/er_200_10/'
        total = len(os.listdir(path))

        for num, filename in enumerate(os.listdir(path)):
            print(filename, num, total)

            graph = nx.read_gpickle(path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            data = Data()

            # Compute map for new nodes in graph.
            var_node = {}
            con_node = {}

            # Number of variables.
            var_i = 0
            # Number of constraints.
            con_i = 0
            # Targets
            y = []
            # Features for variable nodes.
            var_feat = []
            # Feature for constraints nodes.
            con_feat = []
            # Contains right-hand sides of equations.
            rhss = []
            a_sum = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable.
                if node_data['bipartite'] == 0:
                    var_node[i] = var_i
                    var_i += 1

                    y.append(node_data['bias'])
                    # TODO: Scaling meaingful?
                    var_feat.append([node_data['objcoeff'] / 100.0, graph.degree[i]])

                # Node is constraint.
                else:

                    a = []
                    for e in graph.edges(node,data=True):
                        a.append(graph[e[0]][e[1]]['coeff'])
                    a_sum.append(sum(a))

                    con_node[i] = con_i
                    con_i += 1

                    rhs = node_data['rhs']
                    rhss.append(rhs)
                    con_feat.append([rhs, graph.degree[i]])

            num_nodes_var = var_i
            num_nodes_con = con_i
            # Edge list for var->con graphs.
            edge_list_var = []
            # Edge list for con->var graphs.
            edge_list_con = []

            # TODO: Check this.
            edge_features_var = []
            edge_features_con = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
                # Source node is con, target node is var.
                if graph.nodes[s]['bipartite'] == 1:
                    edge_list_con.append([con_node[s], var_node[t]])
                    edge_features_con.append([edge_data['coeff']])
                else:
                    edge_list_var.append([var_node[s], con_node[t]])
                    edge_features_var.append([edge_data['coeff']])

            edge_index_var = torch.tensor(edge_list_var).t().contiguous()
            edge_index_con = torch.tensor(edge_list_con).t().contiguous()

            data.y = torch.from_numpy(np.array(y)).to(torch.float)
            data.edge_index_var = edge_index_var
            data.edge_index_con = edge_index_con
            data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
            data.rhs = torch.from_numpy(np.array(rhss)).to(torch.float)
            data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
            data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)
            data.asums = torch.from_numpy(np.array(a_sum)).to(torch.float)

            data.num_nodes_var = num_nodes_var
            data.num_nodes_con = num_nodes_con

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_var']:
            return torch.tensor([self.num_nodes_var, self.num_nodes_con]).view(2, 1)
        elif key in ['edge_index_con']:
            return torch.tensor([self.num_nodes_con, self.num_nodes_var]).view(2, 1)
        else:
            return 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


# Prepare dadta
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DS')
dataset = GISR(path, transform=MyTransform()).shuffle()
# TODO: log transform.
dataset.data.y = torch.log(dataset.data.y + 1.0)
print(len(dataset))

train_dataset = dataset[0:9000].shuffle()
val_dataset = dataset[8000:9000].shuffle()
test_dataset = dataset[9000:].shuffle()

batch_size = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("### DATA LOADED.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=3, min_lr=0.00001)
print("### SETUP DONE.")


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def train():
    model.train()
    total_loss = 0
    total_loss_mae = 0
    loss = torch.nn.MSELoss()
    rmse = RMSELoss()
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    # mse = torch.nn.SmoothL1Loss()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)

        loss = mse(out, data.y)
        loss.backward()

        total_loss += loss.item() * batch_size
        total_loss_mae += mae(out, data.y).item() * batch_size

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

        error += loss.item() * batch_size

    return error / len(loader.dataset)


best_val_error = None

for epoch in range(1, 500):
    lr = scheduler.optimizer.param_groups[0]['lr']
    mae, loss = train()

    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

    val_error = test(val_loader)

    if best_val_error is None or val_error < best_val_error:
        test_error = test(test_loader)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, loss, mae,
                                                                                               test_error))

# torch.save(model.state_dict(), "train_mip")
