import torch
import torch.nn.functional as F


def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    else:
        raise NotImplementedError('the non_linear_function is not implemented')


def Frechet_mean(manifold, embeddings, weights=None, keepdim=False):
    if weights is None:
        z = torch.sum(embeddings, dim=0, keepdim=True)
    else:
        z = torch.sum(embeddings * weights, dim=0, keepdim=keepdim)
    denorm = manifold.inner(None, z, keepdim=keepdim)
    denorm = denorm.abs().clamp_min(1e-8).sqrt()
    z = z / denorm
    return z