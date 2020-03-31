import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.utils import degree
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform, normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Compute new variable features.
class CONS_TO_VAR(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        # TODO: mean?
        super(CONS_TO_VAR, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_cons = Seq(Lin(in_channels, in_channels - 1), ReLU(), Lin(in_channels - 1, in_channels - 1))
        self.root_vars = Param(torch.Tensor(in_channels, out_channels - 1))
        self.bias = Param(torch.Tensor(out_channels - 1))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.root_vars)
        uniform(size - 1, self.bias)

    def forward(self, hidden_to_var, x, old_vars, edge_index, edge_feature, asums, size):
        # Compute normalization by degree.
        row, _ = edge_index
        deg = degree(row, x[0].size(0), dtype=x[0].dtype)
        deg_inv = deg.pow(-1.0)
        norm = deg_inv[row]

        return self.propagate(hidden_to_var=hidden_to_var, x=x, edge_index=edge_index, size=size, old_vars=old_vars,
                              asums=asums, edge_feature=edge_feature, norm=norm)

    def message(self, hidden_to_var, x_j, x_i, edge_index_j, edge_feature, norm, asums_j):
        # Get coefficients of variable in constraint.
        c = edge_feature[edge_index_j]
        # Get violation of contraint.
        violation = x_j[:, -1]
        violation = hidden_to_var(x_i).view(-1) * violation * c.view(-1)
        # TODO: FIX numerical problems here
        #violation = c.view(-1) / (asums_j) * hidden_to_var(x_i).view(-1) * violation

        # TODO: Scale by coefficient?
        out = self.mlp_cons(c * x_j)

        # Normalize by degree.
        out = norm.view(-1, 1) * torch.cat([out, violation.view(out.size(0), 1)], dim=-1)

        return out

    def update(self, aggr_out, old_vars):
        new_out = torch.empty(aggr_out.size(0), aggr_out.size(1), device=device)

        new_cons = torch.empty(aggr_out.size(0), aggr_out.size(1), device=device)
        new_cons[:, 0:-1] = aggr_out[:, 0:-1] + torch.matmul(old_vars, self.root_vars)

        # TODO: MLP
        new_out[:, -1] = aggr_out[:, -1]
        new_out[:, 0:-1] = F.relu(new_cons[:, 0:-1] + self.bias)

        return new_out


# Compute new variable features.
class VARS_TO_CON(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        # TODO: mean?
        super(VARS_TO_CON, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_var = Seq(Lin(in_channels, in_channels - 1), ReLU(), Lin(in_channels - 1, in_channels - 1))
        self.root_cons = Param(torch.Tensor(in_channels, out_channels - 1))
        self.bias = Param(torch.Tensor(out_channels - 1))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.root_cons)
        uniform(size - 1, self.bias)

    def forward(self, hidden_to_var, x, old_cons, edge_index, edge_feature, rhs, size):
        row, _ = edge_index
        deg = degree(row, x[0].size(0), dtype=x[0].dtype)
        deg_inv = deg.pow(-1.0)
        norm = deg_inv[row]

        return self.propagate(hidden_to_var=hidden_to_var, x=x, edge_index=edge_index, size=size, old_cons=old_cons,
                              edge_feature=edge_feature, rhs=rhs,
                              norm=norm)

    def message(self, hidden_to_var, x_j, x_i, edge_index_j, edge_feature, norm):
        c = edge_feature[edge_index_j]
        # Compute variable assignment.
        var_assign = hidden_to_var(x_j)
        # Variable assignment * coeffient in constraint.
        var_assign = var_assign * c

        # TODO: Scale by coefficient?
        out = norm.view(-1, 1) * self.mlp_var(c * x_j)
        out = torch.cat([out, var_assign], dim=-1)

        return out

    def update(self, aggr_out, old_cons, rhs):
        new_out = torch.empty(aggr_out.size(0), aggr_out.size(1), device=device)

        # Assign violation back to embedding of contraints.
        left = aggr_out[:, -1]
        new_out[:, -1] = left - rhs

        # TODO: MLP

        # New constraint feauture.
        new_cons = torch.empty(aggr_out.size(0), aggr_out.size(1), device=device)
        new_cons[:, 0:-1] = aggr_out[:, 0:-1] + torch.matmul(old_cons, self.root_cons)
        new_out[:, 0:-1] = F.relu(new_cons[:, 0:-1] + self.bias)

        return new_out


class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.var_mlp = Seq(Lin(2, dim - 1), ReLU(), Lin(dim - 1, dim - 1))
        self.con_mlp = Seq(Lin(2, dim - 1), ReLU(), Lin(dim - 1, dim - 1))

        ### TODO: Sigmoid meaningful?
        self.hidden_to_var_1 = Seq(Lin(dim, dim - 1), Sigmoid(), Lin(dim - 1, 1))
        self.v2c_1 = VARS_TO_CON(dim, dim)
        self.c2v_1 = CONS_TO_VAR(dim, dim)

        self.hidden_to_var_2 = Seq(Lin(dim, dim - 1), Sigmoid(), Lin(dim - 1, 1))
        self.v2c_2 = VARS_TO_CON(dim, dim)
        self.c2v_2 = CONS_TO_VAR(dim, dim)

        self.hidden_to_var_3 = Seq(Lin(dim, dim - 1), Sigmoid(), Lin(dim - 1, 1))
        self.v2c_3 = VARS_TO_CON(dim, dim)
        self.c2v_3 = CONS_TO_VAR(dim, dim)

        self.hidden_to_var_4 = Seq(Lin(dim, dim - 1), Sigmoid(), Lin(dim - 1, 1))
        self.v2c_4 = VARS_TO_CON(dim, dim)
        self.c2v_4 = CONS_TO_VAR(dim, dim)

        # Final MLP for regression.
        self.fc1 = Lin(1 * dim, dim)
        self.fc2 = Lin(dim, dim)
        self.fc3 = Lin(dim, dim)
        self.fc4 = Lin(dim, dim)
        self.fc5 = Lin(dim, dim)
        self.fc6 = Lin(dim, 2)

    def forward(self, data):
        if torch.cuda.is_available():
            rand_var = torch.empty(data.var_node_features.size(0), 16).uniform_(0, 1).cuda()
            rand_con = torch.empty(data.con_node_features.size(0), 16).uniform_(0, 1).cuda()
        else:
            rand_var = torch.empty(data.var_node_features.size(0), 16).uniform_(0, 1).cpu()
            rand_con = torch.empty(data.con_node_features.size(0), 16).uniform_(0, 1).cpu()

        if torch.cuda.is_available():
            ones_var = torch.empty(data.var_node_features.size(0), 1).normal_(0, 1).cuda()
            ones_con = torch.empty(data.con_node_features.size(0), 1).normal_(0, 1).cuda()
        else:
            ones_var = torch.zeros(data.var_node_features.size(0), 1).cpu()
            ones_con = torch.zeros(data.con_node_features.size(0), 1).cpu()


        # v = self.con_mlp(torch.cat([rand_var, data.var_node_features], dim=-1))
        # c = self.var_mlp(torch.cat([rand_con, data.con_node_features], dim=-1))

        v = self.con_mlp(data.var_node_features)
        c = self.var_mlp(data.con_node_features)


        v = torch.cat([v, ones_var], dim=-1)
        c = torch.cat([c, ones_con], dim=-1)

        vars = []
        cons = []

        cons.append(
            self.v2c_1(self.hidden_to_var_1, (v, c), c, data.edge_index_var, data.edge_features_var, data.rhs,
                       (data.num_nodes_var.sum(), data.num_nodes_con.sum())))

        vars.append(
            self.c2v_1(self.hidden_to_var_1, (cons[-1], v), v, data.edge_index_con, data.edge_features_con,
                       data.asums, (data.num_nodes_con.sum(), data.num_nodes_var.sum())))

        cons.append(self.v2c_2(self.hidden_to_var_2, (vars[-1], cons[-1]), cons[-1], data.edge_index_var,
                               data.edge_features_var, data.rhs,
                               (data.num_nodes_var.sum(), data.num_nodes_con.sum())))

        vars.append(self.c2v_2(self.hidden_to_var_2, (cons[-1], vars[-1]), vars[-1], data.edge_index_con,
                               data.edge_features_con, data.asums,
                               (data.num_nodes_con.sum(), data.num_nodes_var.sum())))

        cons.append(self.v2c_3(self.hidden_to_var_3, (vars[-1], cons[-1]), cons[-1], data.edge_index_var,
                               data.edge_features_var, data.rhs,
                               (data.num_nodes_var.sum(), data.num_nodes_con.sum())))

        vars.append(self.c2v_3(self.hidden_to_var_3, (cons[-1], vars[-1]), vars[-1], data.edge_index_con,
                               data.edge_features_con, data.asums,
                               (data.num_nodes_con.sum(), data.num_nodes_var.sum())))

        cons.append(self.v2c_4(self.hidden_to_var_4, (vars[-1], cons[-1]), cons[-1], data.edge_index_var,
                               data.edge_features_var, data.rhs,
                               (data.num_nodes_var.sum(), data.num_nodes_con.sum())))

        vars.append(self.c2v_4(self.hidden_to_var_4, (cons[-1], vars[-1]), vars[-1], data.edge_index_con,
                               data.edge_features_con, data.asums,
                               (data.num_nodes_con.sum(), data.num_nodes_var.sum())))


        x = vars[-1]
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        #x = F.sigmoid(self.fc6(x))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x
