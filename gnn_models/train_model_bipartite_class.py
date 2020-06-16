import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
from gnn_models.mip_alternating_arch_class import Net
import torch.nn.functional as F
import sklearn.metrics as metrics


# Dataset and preprocessing.
class GISR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return "tedsfrffrssedrsst"
        return "SET2_class"

    @property
    def processed_file_names(self):
        # return "tessrfffdrdderfdssst"
        return "SET2_class"

    def download(self):
        pass

    def process(self):
        data_list = []

        #path = '../gisp_generator/DATA/er_200_SET2_1k/'
        path = '../gisp_generator/DATA/er_200_SET2_1k/'
        total = len(os.listdir(path))

        for num, filename in enumerate(os.listdir(path)):
            print(filename, num, total)

            # Get graph.
            graph = nx.read_gpickle(path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            data = Data()

            #  Map for new nodes in graph.
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
            # Right-hand sides of equations.
            rhss = []
            # Sums over coefficients.
            a_sum = []
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable.
                if node_data['bipartite'] == 0:
                    var_node[i] = var_i
                    var_i += 1

                    if (node_data['bias'] < 0.05):
                        y.append(0)
                    else:
                        y.append(1)
                    # TODO: Scaling meaingful?
                    var_feat.append([node_data['objcoeff'] / 100.0, graph.degree[i]])

                # Node is constraint.
                else:
                    a = []
                    for e in graph.edges(node, data=True):
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

            data.edge_index_var = edge_index_var
            data.edge_index_con = edge_index_con
            data.y = torch.from_numpy(np.array(y)).to(torch.long)
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


# Prepare data.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DS')
dataset = GISR(path, transform=MyTransform()).shuffle()
len(dataset)


train_index, rest = train_test_split(list(range(0,1000)), test_size=0.2)
val_index = rest[0:100]
test_index = rest[100:]


train_dataset = dataset[train_index].shuffle()
val_dataset = dataset[val_index].shuffle()
test_dataset = dataset[test_index].shuffle()

print(len(val_dataset))

print(1 - test_dataset.data.y.sum().item() / test_dataset.data.y.size(-1))

batch_size = 25
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("### DATA LOADED.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=10,
                                                       min_lr=0.0000001)
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


    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).float().mean().item()
        l += 1

    return correct / l


best_val = 0.0
test_acc = 0.0
for epoch in range(1, 1001):


    train_loss = train(epoch)
    train_acc = test(train_loader)

    val_acc = test(val_loader)
    scheduler.step(val_acc)
    lr = scheduler.optimizer.param_groups[0]['lr']

    if val_acc > best_val:
        best_val = val_acc
        test_acc = test(test_loader)

    # Break if learning rate is smaller 10**-6.
    if lr < 0.000001:
        break

    print(lr)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
