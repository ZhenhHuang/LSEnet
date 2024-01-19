import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from geoopt.manifolds.stereographic.math import mobius_matvec, project, expmap0, mobius_add, logmap0
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import PoincareBall
from torch_scatter import scatter_sum
from torch_geometric.utils import add_self_loops
import math
from utils.utils import gumbel_softmax, adjacency2index, index2adjacency, grad_round


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att)

    def forward(self, x, edge_index):
        h = self.linear(x)
        h = self.agg(h, edge_index)
        return h


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        if self.use_att:
            query = self.query_linear(x)[edge_index[0]]
            key = self.key_linear(x)[edge_index[1]]
            att_adj = 2 + 2 * self.manifold.inner(query, key, dim=-1, keepdim=True)     # (E, 1)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            support_t = scatter_sum(att_adj * x[edge_index[1]], index=edge_index[0], dim=0)
        else:
            support_t = scatter_sum(x[edge_index[1]], index=edge_index[0], dim=0)

        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        return output


class LorentzAtt(nn.Module):
    def __init__(self, manifold, in_features, dropout, return_att=True, nonlin=None):
        super(LorentzAtt, self).__init__()
        self.manifold = manifold
        self.return_att = return_att
        self.in_features = in_features
        self.dropout = dropout
        self.key_linear = LorentzLinear(manifold, in_features, in_features, nonlin=nonlin)
        self.query_linear = LorentzLinear(manifold, in_features, in_features, nonlin=nonlin)
        self.bias = nn.Parameter(torch.zeros(()) + 20)
        self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, res_att=None):
        query = self.query_linear(x)
        key = self.key_linear(x)
        att_adj = 2 + 2 * self.manifold.cinner(query, key)  # (N, N)
        att_adj = att_adj / self.scale + self.bias
        if res_att is not None:
            att_adj = (att_adj + torch.logit(res_att.clamp(1e-5, 0.999))) / 2
        att_adj = torch.sigmoid(att_adj)
        support_t = torch.matmul(att_adj, x)    # (N, N)
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        if self.return_att:
            return output, att_adj
        return output


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_features, num_assign, dropout, bias=False, nonlin=None):
        super(LorentzAssignment, self).__init__()
        # self.assign_linear = LorentzGraphConvolution(manifold, in_features, num_assign, use_att=False,
        #                                              use_bias=bias, dropout=dropout, nonlin=nonlin)
        self.assign_linear = LorentzLinear(manifold, in_features, num_assign,
                                                     bias=bias, dropout=dropout, nonlin=nonlin)

    def forward(self, x, edge_index):
        # ass = self.assign_linear(x, edge_index)
        ass = self.assign_linear(x)
        logits = torch.log_softmax(ass, dim=-1)
        ass = gumbel_softmax(logits, hard=True)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, num_assign, dropout, bias=False, nonlin=None):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        # self.conv = LorentzGraphConvolution(manifold, in_features, out_features, use_att=False,
        #                                              use_bias=bias, dropout=dropout, nonlin=nonlin)
        self.conv = LorentzLinear(manifold, in_features, out_features,
                                                     bias=bias, dropout=dropout, nonlin=None)
        self.assignor = LorentzAssignment(manifold, out_features, num_assign,
                                                     bias=bias, dropout=dropout, nonlin=nonlin)

    def forward(self, x, edge_index):
        x = self.conv(x)
        ass = self.assignor(x, edge_index)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_assigned = support_t / denorm
        adj = index2adjacency(x.shape[0], edge_index)
        adj = ass.t() @ adj @ ass
        edge_index_assigned = adjacency2index(adj)
        return x_assigned, edge_index_assigned, ass