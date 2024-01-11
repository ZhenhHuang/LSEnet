"""LCA construction utils.
    refer to https://github.com/HazyResearch/HypHC/blob/master/utils/lca.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from geoopt.manifolds.stereographic.math import dist0


MIN_NORM = 1e-15


def isometric_transform(a, x):
    """Reflection (circle inversion of x through orthogonal circle centered at a)."""
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
    u = x - a
    return r2 / torch.sum(u ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM) * u + a


def reflection_center(mu):
    """Center of inversion circle."""
    return mu / torch.sum(mu ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)


def euc_reflection(x, a):
    """
    Euclidean reflection (also hyperbolic) of x
    Along the geodesic that goes through a and the origin
    (straight line)
    """
    xTa = torch.sum(x * a, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    proj = xTa * a / norm_a_sq
    return 2 * proj - x


def _halve(x):
    """ computes the point on the geodesic segment from o to x at half the distance """
    return x / (1. + torch.sqrt((1 - torch.sum(x ** 2, dim=-1, keepdim=True)).clamp_min(MIN_NORM))).clamp_min(MIN_NORM)


def hyp_lca(a, b, return_coord=True, proj_hyp=True):
    """
    Computes projection of the origin on the geodesic between a and b, at scale c

    More optimized than hyp_lca1
    """
    r = reflection_center(a)
    b_inv = isometric_transform(r, b)
    o_inv = a
    o_inv_ref = euc_reflection(o_inv, b_inv)
    o_ref = isometric_transform(r, o_inv_ref)
    proj = _halve(o_ref)
    if not return_coord:
        if proj_hyp:
            return dist0(proj, k=torch.tensor([-1.0]).to(proj.device))
        else:
            return proj.norm(p=2, dim=-1)
    else:
        return proj


def hyp_lca2(a, b, return_coord=True, proj_hyp=True):
    a_inv = reflection_center(a)
    b_inv = reflection_center(b)
    c = (a + a_inv) / 2
    d = (b + b_inv) / 2
    i_a = a * torch.tensor([-1, 1]).to(a.device)
    i_b = b * torch.tensor([-1, 1]).to(b.device)
    A = torch.stack([i_a.repeat(b.shape[0], 1, 1),
                     i_b.repeat(1, a.shape[1], 1)], dim=-1).transpose(-1, -2)  # (N, N, 2, 2)
    y = torch.stack([torch.sum(i_a * c, dim=-1).repeat(b.shape[0], 1),
                     torch.sum(i_b * d, dim=-1).repeat(1, a.shape[1])], dim=-1).unsqueeze(-1)   # (N, N, 2)
    x = (torch.pinverse(A) @ y).squeeze(-1)
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    r2 = torch.sum(x ** 2, dim=-1, keepdim=True) - 1
    proj = x / x_norm * (x_norm - r2.abs().sqrt())
    if not return_coord:
        if proj_hyp:
            return dist0(proj, k=torch.tensor([-1.0]).to(proj.device))
        else:
            return proj.norm(p=2, dim=-1)
    else:
        return proj


def equiv_weights(dist2, k, N_0=100, beta=5):
    div = N_0 * np.exp(-beta * k)
    weights = torch.exp(-dist2 / div)
    return weights


# def equiv_weights(dist, radius, k, tau, proj_hyp=False):
#     radius_k = k * radius
#     if proj_hyp:
#         radius_k = 2 * np.arctanh(radius_k)
#     weights = torch.sigmoid((dist - radius_k) / tau)
#     return weights