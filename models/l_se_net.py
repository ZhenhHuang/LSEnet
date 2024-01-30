import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LSENetLayer, LorentzGraphConvolution
from models.encoders import GraphEncoder
from manifold.lorentz import Lorentz
from utils.utils import select_activation
from models.encoders import GraphEncoder
import math
from torch_geometric.utils import dropout_edge


class LSENet(nn.Module):
    def __init__(self, manifold, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=64, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None):
        super(LSENet, self).__init__()
        if max_nums is not None:
            assert len(max_nums) == height - 1, "length of max_nums must equal height-1."
        self.manifold = manifold
        self.nonlin = select_activation(nonlin) if nonlin is not None else None
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.height = height
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
        self.embed_layer = GraphEncoder(self.manifold, 2, in_features + 1, hidden_dim_enc, embed_dim + 1, use_att=False,
                                                     use_bias=True, dropout=dropout, nonlin=self.nonlin)
        self.layers = nn.ModuleList([])
        if max_nums is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            max_nums = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        for i in range(height - 1):
            self.layers.append(LSENetLayer(self.manifold, embed_dim + 1, hidden_features, max_nums[i],
                                           bias=True, use_att=False, dropout=dropout,
                                           nonlin=self.nonlin, temperature=self.temperature))
            # self.num_max = int(self.num_max / decay_rate)

    def forward(self, x, edge_index):

        """mapping x into Lorentz model"""
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        z = self.embed_layer(x, edge_index)
        z = self.normalize(z)

        self.tree_node_coords = {self.height: z}
        self.assignments = {}

        edge = edge_index.clone()
        ass = None
        for i, layer in enumerate(self.layers):
            z, edge, ass = layer(z, edge)
            self.tree_node_coords[self.height - i - 1] = z
            self.assignments[self.height - i] = ass

        self.tree_node_coords[0] = self.manifold.Frechet_mean(z)
        self.assignments[1] = torch.ones(ass.shape[-1], 1).to(x.device)

        return self.tree_node_coords, self.assignments

    def normalize(self, x):
        x = self.manifold.to_poincare(x)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x


