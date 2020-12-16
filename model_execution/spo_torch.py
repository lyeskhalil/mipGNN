import torch
import torch.nn as nn
import cplex
import os
import numpy as np
import argparse


def solveIP(instance_cpx):
    instance_cpx.solve()
    optval = instance_cpx.solution.get_objective_value()
    solution = np.array(instance_cpx.solution.get_values())
    return solution, optval

class SPOLoss(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, coeffs_predictions, sol_true, coeffs_true, instance_cpx):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # optimize with obj = 2*coeff_predicted - coeff_true
        obj_subgradient = (2*coeffs_predictions - coeffs_true) / (torch.max(torch.abs(coeffs_predictions)) + 1)
        instance_cpx.objective.set_linear([(idx, float(obj_subgradient[idx])) for idx in range(len(coeffs_predictions))])
        sol_spo, optval_spo = solveIP(instance_cpx)
        print("fake_optval =", optval_spo)

        print("coeffs_predictions stats: ", 
            torch.mean(coeffs_predictions), 
            torch.median(coeffs_predictions), 
            torch.min(coeffs_predictions), 
            torch.max(coeffs_predictions))

        print("obj_subgradient stats: ", 
            torch.mean(obj_subgradient), 
            torch.median(obj_subgradient), 
            torch.min(obj_subgradient), 
            torch.max(obj_subgradient))
        
        sol_spo = torch.unsqueeze(torch.tensor(sol_spo), 1) 
        objval_predictions_true = torch.dot(sol_spo[:,0], coeffs_true[:,0])
        
        ctx.save_for_backward(
            sol_spo, 
            sol_true, 
            torch.tensor([len(coeffs_predictions),1]))
        
        return torch.tensor([objval_predictions_true])

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        sol_spo, sol_true, coeffs_dim = ctx.saved_tensors
        subgradient = 2*(sol_true - sol_spo)

        # print("^^^^^^", grad_input, '\n', torch.sum(subgradient > 0))

        grad_input = grad_input * subgradient

        return grad_input, None, None, None

class SPONet(torch.nn.Module):
    def __init__(self, num_features, depth, width, relu_sign):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(width, width) for i in range(depth-1)])

        layer0_outdim = 1 if depth == 0 else width
        self.layers.insert(0, nn.Linear(num_features, layer0_outdim))
        if depth > 0:
            self.layers.append(nn.Linear(width, 1, bias=False))
        self.relu_sign = relu_sign

        self.nonlinearity = nn.LeakyReLU()

        print(self.layers)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(torch.t(x))
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)


        return x
        # return self.relu_sign * torch.relu(self.relu_sign * x)
