import cplex
from cplex.callbacks import NodeCallback, BranchCallback

import numpy as np

class branch_empty(BranchCallback):

    def __call__(self):

        for branch_idx in range(self.get_num_branches()):
            self.make_cplex_branch(branch_idx)

class branch_local_exact(BranchCallback):

    def __call__(self):

        if self.is_root:
            self.is_root = False

            self.make_branch(objective_estimate=0.0, constraints=[(self.coeffs, 'L', float(self.threshold))])
            self.make_branch(objective_estimate=0.0, constraints=[(self.coeffs, 'G', float(self.threshold + 1))])


class node_selection(NodeCallback):

    def __call__(self):
        if self.get_num_nodes() == 0:
            return

        # print("NODE SELECTION CALLBACK")

        best_score = -1.0
        best_node = -1

        for node_idx in range(self.get_num_remaining_nodes()):
            node_seqnum = self.get_node_ID(node_idx)
            node_score = self.get_node_data(node_seqnum)
            if node_score > best_score:
                best_node = node_seqnum
                best_score = node_score

        # print("best_score = ", best_score, self.get_depth(best_node))

        self.select_node(best_node)

class branch_attach_data(BranchCallback):

    def __call__(self):
        # print("branching callback")
        nodesel_score = 0.0

        if self.get_num_branches() > 0:
            _, var_info = self.get_branch(0)
            branching_var_idx = var_info[0][0]

        # todo combine estimate with plunging; currently too slow due to jumping around tree
        if self.scoring_function == 'estimate':
            lp_values = self.get_values()
            lp_obj = self.get_objective_value()
            pseudocosts = self.get_pseudo_costs()
            nodesel_score = lp_obj

        # get variable bounds
        lbs = self.get_lower_bounds()
        ubs = self.get_upper_bounds()
        for var_idx in range(self.get_num_cols()):
            if self.scoring_function == 'sum':
                if lbs[var_idx] == ubs[var_idx]:
                    var_val = lbs[var_idx]
                    var_score = self.scores[var_idx] if self.rounding[var_idx] == var_val else 1 - self.scores[var_idx]
                    nodesel_score += var_score
            elif self.scoring_function == 'estimate':
                if lp_values[var_idx] != lbs[var_idx] and lp_values[var_idx] != ubs[var_idx] and var_idx != branching_var_idx:
                    var_score = self.scores[var_idx]
                    nodesel_score += np.min(
                        [pseudocosts[var_idx][0]*var_score, 
                        pseudocosts[var_idx][1]*(1-var_score)])


        for branch_idx in range(self.get_num_branches()):
            node_estimate, var_info = self.get_branch(branch_idx)
            # print(var_info)
            # print(self.rounding[var_idx], var_val, self.rounding[var_idx] == var_val)
            var_idx = var_info[0][0]
            var_val = var_info[0][2]

            # todo cplex might branch on integer variable!
            # assert(lbs[var_idx] != ubs[var_idx])
            # if lbs[var_idx] == ubs[var_idx]:
            #     print(lbs[var_idx], ubs[var_idx], var_idx, var_val, branch_idx, 
            #         self.get_num_branches(), self.get_values(var_idx), self.is_integer_feasible())
                # assert(lbs[var_idx] != ubs[var_idx])

            var_score = 0.0

            if self.scoring_function == 'sum':
                var_score = self.scores[var_idx] if self.rounding[var_idx] == var_val else 1 - self.scores[var_idx]
                nodesel_score_child = (nodesel_score + var_score) / (self.get_current_node_depth() + 1.0)
            elif self.scoring_function == 'estimate':
                var_is_zero = (int(var_val) == 0)
                var_score = pseudocosts[var_idx][int(1-var_is_zero)]*(self.scores[var_idx]*var_is_zero + (1-self.scores[var_idx])*(1-var_is_zero))
                nodesel_score_child = (nodesel_score + var_score)

            self.make_cplex_branch(branch_idx, node_data=nodesel_score_child)
