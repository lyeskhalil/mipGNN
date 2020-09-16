import cplex
from cplex.callbacks import NodeCallback, BranchCallback

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
        # todo incorporate node LP value in score

        # print("branching callback")

        # nodesel_score = self.get_node_data() if self.get_node_ID() > 0 else 0.0

        nodesel_score = 0.0
        # get variable bounds
        lbs = self.get_lower_bounds()
        ubs = self.get_upper_bounds()
        for var_idx in range(self.get_num_cols()):
            if lbs[var_idx] == ubs[var_idx]:
                var_val = lbs[var_idx]
                var_score = self.scores[var_idx] if self.rounding[var_idx] == var_val else 1 - self.scores[var_idx]
                nodesel_score += var_score

        for branch_idx in range(self.get_num_branches()):
            node_estimate, var_info = self.get_branch(branch_idx)
            # print(var_info)
            # print(self.rounding[var_idx], var_val, self.rounding[var_idx] == var_val)
            var_idx = var_info[0][0]
            var_val = var_info[0][2]
            var_score = self.scores[var_idx] if self.rounding[var_idx] == var_val else 1 - self.scores[var_idx]

            # todo factor in other variables fixed at lower or upper bounds
            nodesel_score_child = (nodesel_score + var_score) #/ (self.get_current_node_depth() + 1.0)
            self.make_cplex_branch(branch_idx, node_data=nodesel_score_child)
