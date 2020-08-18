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


# Preprocessing to create Torch dataset.
class GraphDataset(InMemoryDataset):

    def __init__(self, root, data_path, bias_threshold, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_path = data_path
        self.bias_threshold = bias_threshold



    @property
    def raw_file_names(self):
        return "SET2_bi_class"

    @property
    def processed_file_names(self):
        return "SET2_bi_class"

    def download(self):
        pass

    def process(self):
        print("Preprocessing.")

        data_list = []

        path = data_path
        num_graphs = len(os.listdir(path))

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(path)):
            print(filename, num, num_graphs)

            # Get graph.
            graph = nx.read_gpickle(path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            data = Data()

            #  Maps networkx ids to new variable node ids.
            node_to_varnode = {}
            #  Maps networkx ids to new constraint node ids.
            node_to_connode = {}

            # Number of variables.
            num_nodes_var = 0
            # Number of constraints.
            num_nodes_con = 0
            # Targets (classes).
            y = []
            # Features for variable nodes.
            feat_var = []
            # Feature for constraints nodes.
            feat_con = []
            # Right-hand sides of equations.
            feat_rhs = []

            # Iterate over nodes, and collect features.
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable node.
                if node_data['bipartite'] == 0:
                    node_to_varnode[i] = num_nodes_var
                    num_nodes_var += 1

                    if (node_data['bias'] < bias_threshold):
                        y.append(0)
                    else:
                        y.append(1)

                    feat_var.append([node_data['objcoeff'], graph.degree[i]])

                # Node is constraint node.
                elif node_data['bipartite'] == 1:
                    node_to_connode[i] = num_nodes_con
                    num_nodes_con += 1

                    rhs = node_data['rhs']
                    feat_rhs.append(rhs)
                    feat_con.append([rhs, graph.degree[i]])
                else:
                    print("Error in graph format")
                    exit(-1)

            # Edge list for var->con graphs.
            edge_list_var = []
            # Edge list for con->var graphs.
            edge_list_con = []

            # Create features matrices for variable nodes.
            edge_features_var = []
            # Create features matrices for constraint nodes.
            edge_features_con = []

            # TODO: This need to be checked!!!
            # Remark: graph is directed, i.e., each edge exists for each direction.
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
                # Source node is con, target node is var.
                if graph.nodes[s]['bipartite'] == 1:
                    edge_list_con.append([node_to_connode[s], node_to_varnode[t]])
                    edge_features_con.append([edge_data['coeff']])
                else:
                    edge_list_var.append([node_to_varnode[s], node_to_connode[t]])
                    edge_features_var.append([edge_data['coeff']])

            edge_index_var = torch.tensor(edge_list_var).t().contiguous()
            edge_index_con = torch.tensor(edge_list_con).t().contiguous()

            # Create data object.
            data.edge_index_var = edge_index_var
            data.edge_index_con = edge_index_con

            data.y = torch.from_numpy(np.array(y)).to(torch.long)
            data.var_node_features = torch.from_numpy(np.array(feat_var)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(feat_con)).to(torch.float)
            data.rhs = torch.from_numpy(np.array(feat_rhs)).to(torch.float)
            data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
            data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)
            data.num_nodes_var = num_nodes_var
            data.num_nodes_con = num_nodes_con

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Preprocess indices of bipartite graphs to make batching work.
# TODO: Check.
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
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'DS')
# Path to raw graph data.
data_path = '../gisp_generator/DATA/er_200_SET2_1k/'
# Threshold for computing class labels.
bias_threshold = 0.05
# Create dataset.
dataset = GraphDataset(path, data_path, bias_threshold, transform=MyTransform()).shuffle()
len(dataset)


print(dataset[0].is_directed())

# Split data.
train_index, rest = train_test_split(list(range(0, 1000)), test_size=0.2)
val_index = rest[0:100]
test_index = rest[100:]

train_dataset = dataset[train_index].shuffle()
val_dataset = dataset[val_index].shuffle()
test_dataset = dataset[test_index].shuffle()
# TODO: Do not change this.
# np.savetxt("index_er_200_SET2_1k_20", test_index, delimiter=",", fmt="%d")

print(len(val_dataset))
print(len(test_dataset))

print(1 - test_dataset.data.y.sum().item() / test_dataset.data.y.size(-1))

batch_size = 25
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)




#
# print("### DATA LOADED.")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(dim=64).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # Play with this.
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.8, patience=10,
#                                                        min_lr=0.0000001)
# print("### SETUP DONE.")
#
#
# def train(epoch):
#     model.train()
#
#     loss_all = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#
#         loss = F.nll_loss(output, data.y)
#         loss.backward()
#         loss_all += batch_size * loss.item()
#         optimizer.step()
#     return loss_all / len(train_dataset)
#
#
# def test(loader):
#     model.eval()
#
#     correct = 0
#     l = 0
#
#     for data in loader:
#         data = data.to(device)
#         pred = model(data).max(dim=1)[1]
#         correct += pred.eq(data.y).float().mean().item()
#         l += 1
#
#     return correct / l
#
#
# best_val = 0.0
# test_acc = 0.0
# for epoch in range(1, 100):
#
#     train_loss = train(epoch)
#     train_acc = test(train_loader)
#
#     val_acc = test(val_loader)
#     scheduler.step(val_acc)
#     lr = scheduler.optimizer.param_groups[0]['lr']
#
#     if val_acc > best_val:
#         best_val = val_acc
#         test_acc = test(test_loader)
#
#     # Break if learning rate is smaller 10**-6.
#     if lr < 0.000001:
#         break
#
#     print(lr)
#     print('Epoch: {:03d}, Train Loss: {:.7f}, '
#           'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
#                                                        train_acc, test_acc))
#
# torch.save(model.state_dict(), "trained_model_er_200_SET2_1k")
