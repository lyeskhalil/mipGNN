import cplex
from cplex.callbacks import NodeCallback, BranchCallback
from cplex.exceptions import CplexSolverError

import numpy as np
import time
import heapq

class branch_empty(BranchCallback):

    def __call__(self):

        for branch_idx in range(self.get_num_branches()):
            self.make_cplex_branch(branch_idx)

class branch_local_exact(BranchCallback):

    def __call__(self):

        if self.is_root:
            self.is_root = False

            # todo better estimate for L?
            self.make_branch(objective_estimate=0.0, constraints=[(self.coeffs, 'L', float(self.threshold))])
            self.make_branch(objective_estimate=0.0, constraints=[(self.coeffs, 'G', float(self.threshold + 1))])

class node_selection3(NodeCallback):

    def __call__(self):
        if self.get_num_nodes() == 0:
            return
        time_start = time.time()

        self.last_best += 1

        if self.freq_best > 0 and self.last_best % self.freq_best == 0:
            return

        while True:
            try:
                best_score, best_node = heapq.heappop(self.node_priority)
                # print("nodesel", best_score, best_node)
                node_idx = self.get_node_number((best_node,))
            except CplexSolverError as cse:
                continue
            break
        self.select_node(node_idx)

class branch_attach_data2(BranchCallback):

    def __call__(self):
        time_start = time.time()
        # print("branching callback")
        nodesel_score = 0.0

        if self.get_num_branches() > 0:
            _, var_info = self.get_branch(0)
            branching_var_idx = var_info[0][0]

        if self.get_num_nodes() > 0:
            nodesel_score = self.get_node_data()

        for branch_idx in range(self.get_num_branches()):
            node_estimate, var_info = self.get_branch(branch_idx)
            # print(var_info)
            # print(self.rounding[var_idx], var_val, self.rounding[var_idx] == var_val)
            var_idx = var_info[0][0]
            var_val = var_info[0][2]

            var_score = 0.0

            var_score = self.scores[var_idx] if self.rounding[var_idx] == var_val else 1 - self.scores[var_idx]
            var_score *= self.zero_damping if var_val == 0 else 1
            nodesel_score_child = (nodesel_score + var_score) 
            nodesel_score_child_normalized = nodesel_score_child 

            node_seqnum = self.make_cplex_branch(branch_idx, node_data=nodesel_score_child)
            
            heapq.heappush(self.node_priority, (-nodesel_score_child_normalized, node_seqnum))

        self.time += time.time() - time_start

