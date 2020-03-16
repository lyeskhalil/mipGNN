import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx



from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import NNConv

import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform


class RGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_feature, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_feature=edge_feature,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                aggr_out = aggr_out + self.root
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)


class GISR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GISR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ERR"

    @property
    def processed_file_names(self):
        return "ERR"

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
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):

                # Node is a variable.
                if node_data['bipartite'] == 0:
                    y.append(node_data['bias'])
                    node_type.append(0)
                    assoc_var.append(i)
                    coeff = node_data['objcoeff']

                    # TODO: Maybe scale this
                    var_feat.append([coeff / 100.0, graph.degree[i]])
                # Node is constraint.
                else:
                    node_type.append(1)
                    assoc_con.append(i)
                    rhs = node_data['rhs']
                    con_feat.append([rhs, graph.degree[i]])

            y = torch.from_numpy(np.array(y)).to(torch.float).to(torch.float)
            data.y = y
            data.var_node_features = torch.from_numpy(np.array(var_feat)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(con_feat)).to(torch.float)
            data.node_types = torch.from_numpy(np.array(node_type)).to(torch.long)
            data.assoc_var = torch.from_numpy(np.array(assoc_var)).to(torch.long)
            data.assoc_con = torch.from_numpy(np.array(assoc_con)).to(torch.long)

            edge_types = []
            edge_features = []
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):

                if graph.nodes[s]['bipartite']:
                    edge_types.append(0)
                    edge_features.append([edge_data['coeff']])
                else:
                    edge_types.append(0)
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


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))
        self.con_mlp = Seq(Lin(2, dim), ReLU(), Lin(dim, dim))

        self.conv1 = RGCNConv(dim, dim, 2, 1, root_weight=True, bias=True)
        self.conv2 = RGCNConv(dim, dim, 2, 1, root_weight=True, bias=True)
        self.conv3 = RGCNConv(dim, dim, 2, 1, root_weight=True, bias=True)

        # Final MLP for regression.
        self.fc1 = Lin(4 * dim, dim)
        self.fc4 = Lin(dim, 1)

    def forward(self, data):
        n = self.var_mlp(data.var_node_features)
        e = self.con_mlp(data.con_node_features)

        x = e.new_zeros((data.node_types.size(0), n.size(-1)))
        x = x.scatter_(0, data.assoc_var.view(-1, 1).expand_as(n), n)
        x = x.scatter_(0, data.assoc_con.view(-1, 1).expand_as(e), e)

        xs = [x]
        xs.append(F.relu(self.conv1(xs[-1], data.edge_index, data.edge_types, data.edge_features)))
        xs.append(F.relu(self.conv2(xs[-1], data.edge_index, data.edge_types, data.edge_features)))
        xs.append(F.relu(self.conv3(xs[-1], data.edge_index, data.edge_types, data.edge_features)))

        x = torch.cat(xs[0:], dim=-1)
        x = x[data.assoc_var]

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc4(x)

        return x.squeeze(-1)


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
model = Net(dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

print("### SETUP DONE.")

def train():
    model.train()
    total_loss = 0
    total_loss_mae = 0
    mse = torch.nn.MSELoss()
    mse = RMSELoss()
    mae = torch.nn.L1Loss()

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

    val_error = test(val_loader)

    if best_val_error is None or val_error < best_val_error:
        test_error = test(test_loader)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, loss, mae, test_error))

# torch.save(model.state_dict(), "train_mip")

