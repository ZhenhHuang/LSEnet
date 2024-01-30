import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import LorentzGraphConvolution
from utils.utils import select_activation


class GraphEncoder(nn.Module):
    def __init__(self, manifold, n_layers, in_features, n_hidden, out_dim,
                 dropout, nonlin=None, use_att=False, use_bias=False):
        super(GraphEncoder, self).__init__()
        self.manifold = manifold
        self.layers = nn.ModuleList([])
        self.layers.append(LorentzGraphConvolution(self.manifold, in_features,
                                                   n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=None))
        for i in range(n_layers - 2):
            self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))
        self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       out_dim, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = torch.sigmoid((self.r - dist) / self.t)
        return probs
