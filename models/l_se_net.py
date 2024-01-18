import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LSENetLayer, LorentzGraphConvolution
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
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
        self.embed_layer = LorentzGraphConvolution(self.manifold, in_features + 1, embed_dim + 1, use_att=False,
                                                     use_bias=False, dropout=dropout, nonlin=self.nonlin)
        self.layers = nn.ModuleList([])
        coeff = int(np.exp(np.log(num_nodes) / height))
        self.num_max = int(num_nodes / coeff)
        for _ in range(height):
            self.layers.append(LSENetLayer(self.manifold, embed_dim + 1, embed_dim + 1, self.num_max,
                                           bias=False, dropout=dropout, nonlin=self.nonlin))

    def forward(self, x, edge_index):

        """mapping x into Lorentz model"""
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        z = self.embed_layer(x, edge_index)
        z = self.normalize(z)

        self.tree_node_coords = {self.height: z}
        self.assignments = {self.height: torch.eye(self.num_nodes).to(x.device)}

        edge = edge_index.clone()
        prev_num = self.num_max
        for i, layer in enumerate(self.layers):
            z, edge, ass, prev_num = layer(z, edge, prev_num)
            if i == len(self.layers) - 1:
                z = self.manifold.Frechet_mean(z)
            self.tree_node_coords[self.height - i - 1] = z
            self.assignments[self.height - i - 1] = ass

        return self.tree_node_coords, self.assignments

    def normalize(self, x):
        x = self.manifold.to_poincare(x)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x


