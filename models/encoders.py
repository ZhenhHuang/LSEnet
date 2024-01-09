import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import LorentzGraphConvolution
from utils.utils import select_activation


class GraphEncoder(nn.Module):
    def __init__(self, manifold, n_layers, in_features, n_hidden, out_dim, dropout, activation='relu', use_att=False):
        super(GraphEncoder, self).__init__()
        act = select_activation(activation)
        self.manifold = manifold
        self.layers = nn.ModuleList([])
        self.layers.append(LorentzGraphConvolution(self.manifold, in_features,
                                                   n_hidden, False, dropout=dropout, use_att=use_att))
        for i in range(n_layers - 2):
            self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       n_hidden, False, dropout=dropout, use_att=use_att, nonlin=act))
        self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       out_dim + 1, False, dropout=dropout, use_att=use_att, nonlin=act))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
