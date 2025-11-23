'''
Taken from https://github.com/mmany/pytorch-GDL/tree/main
'''
import torch.nn as nn

class GradientDifferenceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, inputs, targets):

        gradient_diff = (inputs.diff(axis=0)-targets.diff(axis=0)).pow(2) + (inputs.diff(axis=1)-targets.diff(axis=1)).pow(2)
        loss_gdl = gradient_diff.sum()/inputs.numel()

        return loss_gdl

class MSE_and_2D_GDL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSE_and_2D_GDL, self).__init__()

    def forward(self, inputs, targets, lambda_mse, lambda_gdl):

        squared_error = (inputs - targets).pow(2)
        gradient_diff_i = (inputs.diff(axis=-1)-targets.diff(axis=-1)).pow(2)
        gradient_diff_j = (inputs.diff(axis=-2)-targets.diff(axis=-2)).pow(2)
        loss = (lambda_mse*squared_error.sum() + lambda_gdl*gradient_diff_i.sum() + lambda_gdl*gradient_diff_j.sum() )/inputs.numel()

        return loss

class MSE_and_3D_GDL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSE_and_3D_GDL, self).__init__()

    def forward(self, inputs, targets, lambda_mse, lambda_gdl):

        squared_error = (inputs - targets).pow(2)
        gradient_diff_i = (inputs.diff(axis=-1)-targets.diff(axis=-1)).pow(2)
        gradient_diff_j = (inputs.diff(axis=-2)-targets.diff(axis=-2)).pow(2)
        gradient_diff_d = (inputs.diff(axis=-3)-targets.diff(axis=-3)).pow(2)
        loss = (lambda_mse*squared_error.sum() + lambda_gdl*gradient_diff_i.sum() + lambda_gdl*gradient_diff_j.sum() + lambda_gdl*gradient_diff_d.sum())/inputs.numel()

        return loss