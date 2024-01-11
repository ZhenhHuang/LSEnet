import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LorentzGraphConvolution, LorentzLinear, LorentzAgg, LorentzAtt
from manifold.lorentz import Lorentz
from utils.utils import select_activation
from models.encoders import GraphEncoder
import math


class LSENet(nn.Module):
    def __init__(self, manifold, in_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=64, dropout=0.5, nonlin='relu'):
        super(LSENet, self).__init__()
        self.manifold = manifold
        self.nonlin = select_activation(nonlin)
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.height = height
        self.layers = nn.ModuleList([])
        # self.layers.append(LorentzGraphConvolution(self.manifold, in_features + 1, embed_dim + 1,
        #                                            use_bias=False, use_att=True, nonlin=None, dropout=dropout))
        self.layers.append(GraphEncoder(self.manifold, 2, in_features + 1, 256, embed_dim + 1,
                                        dropout, nonlin, use_att=False))
        for _ in range(height - 1):
            self.layers.append(LorentzAtt(self.manifold, embed_dim + 1, dropout=dropout, return_att=True))

    def forward(self, x, edge_index):

        """mapping x into Lorentz model"""
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)

        self.embeddings = []
        self.weights = [torch.eye(self.num_nodes).to(x.device)]

        z = self.layers[0](x, edge_index)
        self.embeddings.append(z)

        for i, layer in enumerate(self.layers[1:]):
            z, att = layer(z, self.weights[i])
            self.embeddings.append(z)
            self.weights.append(att)

        z = torch.sum(z, dim=0, keepdim=True)

        denorm = (-self.manifold.inner(None, z, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        self.embeddings.append(z)
        self.weights.append(torch.ones(self.num_nodes, self.num_nodes).to(x.device))
        return self.embeddings[::-1], self.weights[::-1]



