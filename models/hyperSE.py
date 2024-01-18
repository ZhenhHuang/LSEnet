import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import project, dist0, dist
import numpy as np
from utils.lca import hyp_lca, equiv_weights
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LorentzGraphConvolution, LorentzLinear
from manifold.lorentz import Lorentz
from manifold.poincare import Poincare
from models.encoders import GraphEncoder
import math
from models.l_se_net import LSENet


MIN_NORM = 1e-15


class HyperSE(nn.Module):
    def __init__(self, in_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=2, dropout=0.1, nonlin='relu', max_size=0.999):
        super(HyperSE, self).__init__()
        init_size = 1e-2
        self.k = torch.tensor([-1.0])
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.manifold = Lorentz()
        self.scale = nn.Parameter(torch.tensor([init_size]), requires_grad=True)
        self.init_size = init_size
        self.min_size = 1e-2
        self.max_size = max_size
        self.encoder = LSENet(self.manifold, in_features, num_nodes, height, temperature, embed_dim, dropout, nonlin)

    def forward(self, data, device=torch.device('cuda:0')):
        features = data['feature'].to(device)
        edge_index = data['edge_index'].to(device)
        embeddings, assignments = self.encoder(features, edge_index)
        self.disk_embeddings = {}
        for height, x in embeddings.items():
            x = self.manifold.to_poincare(x)
            # x = self.normalize(x)
            x = project(x, k=self.k.to(x.device), eps=MIN_NORM)
            self.disk_embeddings[height] = x
        ind_pairs = {self.height: assignments[self.height]}
        temp = assignments[self.height]
        for k in range(self.height - 1, -1, -1):
            temp = temp @ assignments[k]
            ind_pairs[k] = temp @ temp.t()
        self.ind_pairs = ind_pairs
        return self.disk_embeddings[self.height]

    def normalize(self, embeddings):
        min_size = self.min_size
        max_size = self.max_size
        embeddings_normed = F.normalize(embeddings, p=2, dim=-1) * 0.999
        return embeddings_normed

    def loss(self, data, device=torch.device('cuda:0')):
        """_summary_

        Args:
            data: dict
            device: torch.Device
        """
        weight = data['weight'].to(device)
        edge_index = data['edge_index'].to(device)
        degrees = data['degrees'].to(device)
        features = data['feature'].to(device)

        embeddings, assignments = self.encoder(features, edge_index)

        loss = 0
        vol_G = weight.sum()
        ind_pairs = {self.height: assignments[self.height]}
        temp = assignments[self.height]
        for k in range(self.height - 1, -1, -1):
            temp = temp @ assignments[k]
            ind_pairs[k] = temp @ temp.t()

        for k in range(1, self.height + 1):
            log_sum_dl_k = torch.log2(torch.sum(ind_pairs[k] * degrees, -1))  # (N, )
            log_sum_dl_k_1 = torch.log2(torch.sum(ind_pairs[k - 1] * degrees, -1))  # (N, )
            ind_i_j = ind_pairs[k][edge_index[0], edge_index[1]]
            weight_sum = scatter_sum(ind_i_j * weight, index=edge_index[0])  # (N, )
            d_log_sum_k = (degrees - weight_sum) * log_sum_dl_k  # (N, )
            d_log_sum_k_1 = (degrees - weight_sum) * log_sum_dl_k_1  # (N, )
            loss += torch.sum(d_log_sum_k - d_log_sum_k_1)
        loss = -1 / vol_G * loss + Poincare().dist0(self.manifold.to_poincare(embeddings[0]))

        neg_edge_index = data['neg_edge_index'].to(device)
        edges = torch.concat([edge_index, neg_edge_index], dim=-1)
        prob = self.manifold.dist(embeddings[self.height][edges[0]], embeddings[self.height][edges[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.concat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(device)
        loss += F.binary_cross_entropy(prob, label)
        return loss
