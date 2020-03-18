import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.utils import degree

import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform, normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MIPGNN(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(MIPGNN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Maps variable embedding to a scalar variable assignmnet.
        # TODO: Sigmoid?
        self.hidden_to_var = Seq(Lin(in_channels, in_channels - 1), ReLU(), Lin(in_channels - 1, 1))

        self.mlp_cons = Seq(Lin(in_channels-1, in_channels - 1), ReLU(), Lin(in_channels - 1, in_channels - 1))
        self.mlp_vars = Seq(Lin(in_channels-1, in_channels - 1), ReLU(), Lin(in_channels - 1, in_channels - 1))

        self.w_cons = Param(torch.Tensor(in_channels - 1, out_channels - 1))
        self.w_vars = Param(torch.Tensor(in_channels - 1, out_channels - 1))

        self.root_cons = Param(torch.Tensor(in_channels, out_channels))
        self.root_vars = Param(torch.Tensor(in_channels, out_channels))
        self.bias = Param(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size - 1, self.w_cons)
        uniform(size - 1, self.w_vars)
        uniform(size, self.root_cons)
        uniform(size, self.root_vars)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_feature, assoc_con, assoc_var, rhs, size=None):
        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1.0)
        norm = deg_inv[row]

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_feature=edge_feature,
                              assoc_con=assoc_con, assoc_var=assoc_var, rhs=rhs, norm=norm)

    def message(self, x_j, edge_index_j, edge_type, edge_feature, norm):
        # Split data.
        # Variable nodes.
        x_j_0 = x_j[edge_type == 0]
        # Constraint nodes.
        x_j_1 = x_j[edge_type == 1]

        ### Vars -> Cons.
        #  x_j is a variable node.
        c = edge_feature[edge_index_j][edge_type == 0]
        # Compute variable assignment.
        var_assign = self.hidden_to_var(x_j_0)
        # Variable assignment * coeffient in constraint.
        var_assign = var_assign * c
        # TODO: Scale by coefficient?
        # out_0 = norm.view(-1, 1)[edge_type == 0] * torch.matmul(x_j_0[:, 0:-1], self.w_cons)
        out_0 = norm.view(-1, 1)[edge_type == 0] * self.mlp_cons(x_j_0[:, 0:-1])
        # Assign left side of constraint to last column.
        out_0 = torch.cat([out_0, var_assign], dim=-1)

        ### Cons -> Vars.
        c = edge_feature[edge_index_j][edge_type == 1]
        # Get violation of contraint.
        violation = x_j_1[:, -1]
        # TODO: This should be scaled. Multiplied by current a
        violation = c.view(-1) * violation
        # TODO: Scale by coefficient?
        # out_1 = torch.matmul(x_j_1[:, 0:-1], self.w_vars)
        out_1 = self.mlp_vars(x_j_1[:, 0:-1])
        out_1 = norm.view(-1, 1)[edge_type == 1] * torch.cat([out_1, violation.view(-1, 1)], dim=-1)

        new_out = torch.zeros(x_j.size(0), self.out_channels, device=device)
        new_out[edge_type == 0] = out_0
        new_out[edge_type == 1] = out_1

        return new_out

    def update(self, aggr_out, x, assoc_con, assoc_var, rhs):
        new_out = torch.zeros(aggr_out.size(0), aggr_out.size(1), device=device)

        # Assign violation back to embedding of contraints.
        t = aggr_out[assoc_con, -1]
        new_out[assoc_con, -1] = t - rhs
        new_out[assoc_con, 0:-1] = aggr_out[assoc_con, 0:-1]

        new_out[assoc_var] = aggr_out[assoc_var]

        # Normalize aggregation.
        # TODO: Implement normalization!!!

        t_1 = new_out[assoc_con] + torch.matmul(x[assoc_con], self.root_vars)
        t_2 = new_out[assoc_var] + torch.matmul(x[assoc_var], self.root_vars)

        out = torch.zeros(new_out.size(0), new_out.size(1), device=device)
        out[assoc_con] = t_1
        out[assoc_var] = t_2

        new_out = new_out + self.bias

        return new_out


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim - 3), ReLU(), Lin(dim - 3, dim - 3))
        self.con_mlp = Seq(Lin(2, dim - 3), ReLU(), Lin(dim - 3, dim - 3))

        self.conv1 = MIPGNN(dim, dim)
        self.conv2 = MIPGNN(dim, dim)
        self.conv3 = MIPGNN(dim, dim)
        self.conv4 = MIPGNN(dim, dim)
        self.conv5 = MIPGNN(dim, dim)
        self.conv6 = MIPGNN(dim, dim)

        # Final MLP for regression.
        self.fc1 = Lin(7 * dim, dim)
        self.fc2 = Lin(dim, dim)
        self.fc3 = Lin(dim, dim)
        self.fc4 = Lin(dim, 1)

    def forward(self, data):
        if torch.cuda.is_available():
            ones_var = torch.zeros(data.var_node_features.size(0), 1).cuda()
            ones_con = torch.zeros(data.con_node_features.size(0), 1).cuda()
        else:
            ones_var = torch.zeros(data.var_node_features.size(0), 1).cpu()
            ones_con = torch.zeros(data.con_node_features.size(0), 1).cpu()

        n = torch.cat([self.var_mlp(data.var_node_features), data.var_node_features, ones_var], dim=-1)
        e = torch.cat([self.con_mlp(data.con_node_features), data.con_node_features, ones_con], dim=-1)

        # Merge node features together.
        x = e.new_zeros((data.node_types.size(0), n.size(-1)))
        x = x.scatter_(0, data.assoc_var.view(-1, 1).expand_as(n), n)
        x = x.scatter_(0, data.assoc_con.view(-1, 1).expand_as(e), e)

        xs = [x]
        xs.append(F.relu(
            self.conv1(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var,
                       data.rhs)))
        xs.append(F.relu(
            self.conv2(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var,
                       data.rhs)))
        xs.append(F.relu(
            self.conv3(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var,
                       data.rhs)))
        xs.append(F.relu(
            self.conv4(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var,
                       data.rhs)))
        xs.append(F.relu(
            self.conv5(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var,
                       data.rhs)))
        xs.append(F.relu(
            self.conv6(xs[-1], data.edge_index, data.edge_types, data.edge_features, data.assoc_con, data.assoc_var,
                       data.rhs)))

        x = torch.cat(xs[0:], dim=-1)
        x = x[data.assoc_var]

        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        # x = F.dropout(x, p=0.5, training=self.training)

        # TODO: Sigmoid meaningful?
        # x = F.sigmoid(self.fc4(x))
        x = self.fc4(x)

        return x.squeeze(-1)
