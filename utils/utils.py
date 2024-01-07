import torch
import torch.nn.functional as F


def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    else:
        raise NotImplementedError('the non_linear_function is not implemented')