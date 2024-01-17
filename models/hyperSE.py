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
from models.encoders import GraphEncoder
import math
from models.l_se_net import LSENet


MIN_NORM = 1e-15


# class HyperSE(nn.Module):
#     def __init__(self, in_features, num_nodes, d_hyp=16, height=2, temperature=0.1, c=0.5,
#                  embed_dim=64, min_size=1e-2, max_size=0.999, dropout=0.5, use_att=False):
#         super(HyperSE, self).__init__()
#         init_size = 1e-2
#         self.k = torch.tensor([-1.0])
#         self.num_nodes = num_nodes
#         self.height = height
#         self.tau = temperature
#         self.manifold = Lorentz()
#         self.encoder = GraphEncoder(self.manifold, 2, in_features + 1, 512, embed_dim + 1,
#                                     dropout, 'relu', use_att=use_att)
#         self.proj = LorentzLinear(self.manifold, embed_dim + 1, d_hyp + 1, bias=False, dropout=0.1)
#         self.scale = nn.Parameter(torch.tensor([init_size]), requires_grad=True)
#         self.c = max_size / (height + 1)
#         self.init_size = init_size
#         self.min_size = height * self.c
#         self.max_size = max_size
#
#     def forward(self, data, device=torch.device('cuda:0')):
#         features = data['feature'].to(device)
#         o = torch.zeros_like(features).to(device)
#         features = torch.cat([o[:, 0:1], features], dim=1)
#         features = self.manifold.expmap0(features)
#         edge_index = data['edge_index'].to(device)
#         embedding_l = self.proj(self.encoder(features, edge_index))
#         embedding = self.manifold.to_poincare(embedding_l)
#         embedding = self.normalize(embedding)
#         embedding = project(embedding, k=self.k.to(embedding.device), eps=MIN_NORM)
#         return embedding
#
#     def normalize(self, embeddings):
#         min_size = self.min_size
#         max_size = self.max_size
#         embeddings_normed = F.normalize(embeddings, p=2, dim=-1)
#         return embeddings_normed * self.scale.clamp(min_size, max_size)
#
#     def loss(self, data, device=torch.device('cuda:0')):
#         """_summary_
#
#         Args:
#             data: dict
#             device: torch.Device
#         """
#         weight = data['weight'].to(device)
#         edge_index = data['edge_index'].to(device)
#         degrees = data['degrees'].to(device)
#         features = data['feature'].to(device)
#
#         o = torch.zeros_like(features).to(device)
#         features = torch.cat([o[:, 0:1], features], dim=1)
#         features = self.manifold.expmap0(features)
#         embedding_l = self.proj(self.encoder(features, edge_index))
#         embedding = self.manifold.to_poincare(embedding_l)
#
#         se_loss = self.calc_se_loss(embedding, edge_index, weight, degrees)
#         # cl_loss = self.calc_contrastive_loss(embedding_l)
#         loss = se_loss
#         # print(se_loss.item(), cl_loss.item())
#         return loss
#
#     def calc_se_loss(self, embedding, edge_index, weight, degrees):
#         device = embedding.device
#         loss = 0
#         vol_G = weight.sum()
#
#         embedding = self.normalize(embedding)
#         # dist_pairs = hyp_lca(embedding[None], embedding[:, None, :] + MIN_NORM, return_coord=False,
#         #                      proj_hyp=False)  # Euclidean circle
#         # dist_pairs = -(2 + 2 * self.manifold.cinner(embedding, embedding))
#         dist_pairs = dist(embedding[None], embedding[:, None], k=torch.tensor([-1.0]).to(device))
#         # ind_pairs = [torch.ones(self.num_nodes, self.num_nodes).to(device)]
#         ind_pairs = []
#         for k in range(0, self.height + 1):
#             ind_pairs_k = equiv_weights(dist_pairs, k, N_0=50, beta=2)
#             ind_pairs.append(ind_pairs_k)
#         # ind_pairs.append(torch.eye(self.num_nodes).to(device))
#         self.ind_pairs = ind_pairs
#         for k in range(1, self.height + 1):
#             log_sum_dl_k = torch.log2(torch.sum(ind_pairs[k] * degrees, -1))  # (N, )
#             log_sum_dl_k_1 = torch.log2(torch.sum(ind_pairs[k - 1] * degrees, -1))  # (N, )
#             ind_i_j = ind_pairs[k][edge_index[0], edge_index[1]]
#             weight_sum = scatter_sum(ind_i_j * weight, index=edge_index[0])  # (N, )
#             d_log_sum_k = (degrees - weight_sum) * log_sum_dl_k  # (N, )
#             d_log_sum_k_1 = (degrees - weight_sum) * log_sum_dl_k_1  # (N, )
#             loss += torch.sum(d_log_sum_k - d_log_sum_k_1)
#         loss = -1 / vol_G * loss
#         for k in range(1, self.height + 1):
#             vol_k = torch.log2(torch.sum(ind_pairs[k] * degrees, -1))
#             vol_k_1 = torch.log2(torch.sum(ind_pairs[k - 1] * degrees, -1))
#             loss += torch.mean(torch.sigmoid(5. - (vol_k - vol_k_1)) / self.tau)
#         return loss
#
#     def calc_contrastive_loss(self, z, temperature=0.1):
#         norm = z.norm(dim=-1).clamp_min(1e-8)
#         sim = torch.einsum('ik,jk->ij', z, z) / (torch.einsum('i,j->ij', norm, norm) + MIN_NORM)
#         sim = torch.exp(sim / temperature)
#         div = sim.sum(dim=-1).clamp_min(1e-8)
#         loss = 0
#         for k in range(1, self.height + 1):
#             info = torch.sum(sim * self.ind_pairs[k], dim=-1) / div
#             loss += -torch.log(info).mean()
#         return loss
#
#     def decode(self):
#         L_nodes = [i for i in range(self.num_nodes)]
#         embeddings = self.enc_layers(torch.arange(self.num_nodes)).detach().cpu().numpy()
#         return construct_tree(L_nodes, embeddings, self.height, self.c, k=1)


class HyperSE(nn.Module):
    def __init__(self, in_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=64, dropout=0.1, nonlin='relu', max_size=0.999):
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
            x = self.normalize(x)
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
        embeddings_normed = F.normalize(embeddings, p=2, dim=-1)
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
        loss = -1 / vol_G * loss + self.manifold.dist0(embeddings[0])

        # neg_edge_index = data['neg_edge_index'].to(device)
        # edges = torch.concat([edge_index, neg_edge_index], dim=-1)
        # prob = self.manifold.dist(embeddings[-1][edges[0]], embeddings[-1][edges[1]])
        # prob = torch.sigmoid((2. - prob) / 1.)
        # label = torch.concat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(device)
        # loss += F.binary_cross_entropy(prob, label)
        return loss
