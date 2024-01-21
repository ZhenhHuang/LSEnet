import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import project, dist0, dist
import numpy as np
from utils.utils import gumbel_softmax
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LorentzGraphConvolution, LorentzLinear
from manifold.lorentz import Lorentz
from manifold.poincare import Poincare
from models.encoders import GraphEncoder
import math
from models.l_se_net import LSENet


MIN_NORM = 1e-15
EPS = 1e-6


class HyperSE(nn.Module):
    def __init__(self, in_features, hidden_features, num_nodes, height=3, temperature=0.2,
                 embed_dim=2, dropout=0.5, nonlin='relu'):
        super(HyperSE, self).__init__()
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.manifold = Lorentz()
        self.encoder = LSENet(self.manifold, in_features, hidden_features, num_nodes, height, temperature, embed_dim, dropout, nonlin)

    def forward(self, data, device=torch.device('cuda:0')):
        features = data['feature'].to(device)
        edge_index = data['edge_index'].to(device)
        embeddings, clu_mat = self.encoder(features, edge_index)
        self.disk_embeddings = {}
        for height, x in embeddings.items():
            x = self.manifold.to_poincare(x)
            self.disk_embeddings[height] = x.detach()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(device)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
        for k, v in ass_mat.items():
            ass_mat[k] = gumbel_softmax(v.log(), temperature=self.tau, hard=True)
        self.ass_mat = ass_mat
        return self.disk_embeddings[self.height]

    def loss(self, data, device=torch.device('cuda:0')):
        """_summary_

        Args:
            data: dict
            device: torch.Device
        """
        weight = data['weight']
        edge_index = data['edge_index']
        degrees = data['degrees']
        features = data['feature']

        embeddings, clu_mat = self.encoder(features, edge_index)

        loss = 0
        vol_G = weight.sum()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(device)}
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
            vol_dict[k] = torch.einsum('ij, i->j', ass_mat[k], degrees)

        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', clu_mat[k], vol_dict[k - 1])  # (N_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (N_k, )
            ass_i = ass_mat[k][edge_index[0]]   # (E, N_k)
            ass_j = ass_mat[k][edge_index[1]]
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # (N_k, )
            delta_vol = vol_dict[k] - weight_sum    # (N_k, )
            loss += torch.sum(delta_vol * log_vol_ratio_k)
        loss = -1 / vol_G * loss + 5 * Poincare().dist0(self.manifold.to_poincare(embeddings[0]))

        neg_edge_index = data['neg_edge_index'].to(device)
        edges = torch.concat([edge_index, neg_edge_index], dim=-1)
        prob = self.manifold.dist(embeddings[self.height][edges[0]], embeddings[self.height][edges[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.concat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(device)
        loss += F.binary_cross_entropy(prob, label)
        return loss