import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import project, dist0
import numpy as np
from utils.lca import hyp_lca, equiv_weights
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LorentzGraphConvolution
from manifold.lorentz import Lorentz
from models.encoders import GraphEncoder
from torch_geometric.utils import dropout_edge, mask_feature


MIN_NORM = 1e-15


class HyperSE(nn.Module):
    def __init__(self, in_features, num_nodes, d_hyp=2, height=2, temperature=0.1, c=0.5,
                 init_size=1e-3, min_size=1e-2, max_size=0.999):
        super(HyperSE, self).__init__()
        init_size = 1.0
        self.k = torch.tensor([-1.0])
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.manifold = Lorentz()
        self.encoder = GraphEncoder(self.manifold, 2, in_features, 256, d_hyp, 0.1, 'relu')
        self.scale = nn.Parameter(torch.tensor([init_size]), requires_grad=True)
        self.c = max_size / (height + 1)
        self.init_size = init_size
        self.min_size = height * self.c
        self.max_size = max_size
    
    def forward(self, data, device=torch.device('cuda:0')):
        features = data['feature'].to(device)
        edge_index = data['edge_index'].to(device)
        embedding_l = self.encoder(features, edge_index)
        # embedding_l = self.manifold.projx(embedding_l)
        embedding = self.manifold.to_poincare(embedding_l)
        embedding = self.normalize(embedding)
        embedding = project(embedding, k=self.k.to(embedding.device), eps=MIN_NORM)
        return embedding
        
    def normalize(self, embeddings):
        min_size = self.min_size
        max_size = self.max_size
        embeddings_normed = F.normalize(embeddings, p=2, dim=-1)
        return embeddings_normed * self.scale.clamp(min_size, max_size)
    
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
        loss = 0

        embedding_l = self.encoder(features, edge_index)
        # embedding_l = self.manifold.projx(embedding_l)
        embedding = self.manifold.to_poincare(embedding_l)
        embedding = self.normalize(embedding)

        neg_edge, mask = dropout_edge(edge_index, p=0.5)
        neg_feat, _ = mask_feature(features, p=0.5)
        embedding_l_neg = self.encoder(neg_feat, neg_edge)
        cl_loss = self.calc_contrastive_loss(embedding_l, embedding_l_neg)

        se_loss = self.calc_se_loss(embedding, edge_index, weight, degrees)
        loss += se_loss + cl_loss
        return loss

    def calc_se_loss(self, embedding, edge_index, weight, degrees):
        device = embedding.device
        loss = 0
        vol_G = weight.sum()
        dist_pairs = hyp_lca(embedding[None], embedding[:, None, :] + MIN_NORM, return_coord=False,
                             proj_hyp=False)  # Euclidean circle
        ind_pairs = [torch.ones(self.num_nodes, self.num_nodes).to(device)]
        for k in range(1, self.height):
            ind_pairs_k = equiv_weights(dist_pairs, self.c, k, self.tau, proj_hyp=False)
            ind_pairs.append(ind_pairs_k)
        ind_pairs.append(torch.eye(self.num_nodes).to(device))
        for k in range(1, self.height + 1):
            log_sum_dl_k = torch.log2(torch.sum(ind_pairs[k] * degrees, -1))  # (N, )
            log_sum_dl_k_1 = torch.log2(torch.sum(ind_pairs[k - 1] * degrees, -1))  # (N, )
            ind_i_j = ind_pairs[k][edge_index[0], edge_index[1]]
            weight_sum = scatter_sum(ind_i_j * weight, index=edge_index[0])  # (N, )
            d_log_sum_k = (degrees - weight_sum) * log_sum_dl_k  # (N, )
            d_log_sum_k_1 = (degrees - weight_sum) * log_sum_dl_k_1  # (N, )
            loss += torch.sum(d_log_sum_k - d_log_sum_k_1)
        loss = -1 / vol_G * loss
        return loss

    def calc_contrastive_loss(self, pos, neg, temperature=0.1):
        norm1 = pos.norm(dim=-1).clamp_min(1e-8)
        norm2 = neg.norm(dim=-1).clamp_min(1e-8)
        sim_matrix = torch.einsum('ik,jk->ij', pos, neg) / (torch.einsum('i,j->ij', norm1, norm2) + MIN_NORM)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) + MIN_NORM)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) + MIN_NORM)

        loss_1 = -torch.log(loss_1.clamp_min(1e-8)).mean()
        loss_2 = -torch.log(loss_2.clamp_min(1e-8)).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss

    def decode(self):
        L_nodes = [i for i in range(self.num_nodes)]
        embeddings = self.enc_layers(torch.arange(self.num_nodes)).detach().cpu().numpy()
        return construct_tree(L_nodes, embeddings, self.height, self.c, k=1)