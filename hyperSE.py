import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import project
import numpy as np
from lca import hyp_lca
from torch_scatter import scatter_sum


MIN_NORM = 1e-15


class HyperSE(nn.Module):
    def __init__(self, num_nodes, d_hyp=2, temperature=0.1, c=0.5, 
                 init_size=1.0, min_size=1e-2, max_size=0.999):
        super(HyperSE, self).__init__()
        self.k = torch.tensor([-1.0])
        self.num_nodes = num_nodes
        self.tau = temperature
        self.embeddings = nn.Embedding(num_nodes, embedding_dim=d_hyp)
        self.scale = nn.Parameter(torch.tensor([init_size]), requires_grad=True)
        self.embeddings.weight.data = project(
            self.scale * (2 * torch.rand(num_nodes, d_hyp) - 1), 
            k=self.k, eps=MIN_NORM)
        self.c = c
        self.init_size = init_size
        self.min_size = min_size
        self.max_size = max_size
    
    def forward(self):
        embedding = self.normalize(self.embeddings.weight.data)
        embedding = project(embedding, k=self.k.to(embedding.device), eps=MIN_NORM)
        return embedding
        
    def normalize(self, embeddings):
        min_size = self.min_size
        max_size = self.max_size
        embeddings_normed = F.normalize(embeddings, p=2, dim=-1)
        return embeddings_normed * self.scale.clamp(min_size, max_size)
    
    def loss(self, edge_index, degrees, weight=None, device=torch.device('cuda:0')):
        """_summary_

        Args:
            edge_index (_type_): 2, E
            degrees (_type_): N
            weight (): E
        """
        if weight is None:
            weight = torch.ones(edge_index.shape[-1])
        weight = weight.to(device)
        edge_index = edge_index.to(device)
        degrees = degrees.to(device)
        
        embedding_i = self.embeddings(edge_index[0])    # (E, d)
        embedding_j = self.embeddings(edge_index[1])
        embedding_i = self.normalize(embedding_i)
        embedding_j = self.normalize(embedding_j)
        
        dist_ivj = hyp_lca(embedding_i, embedding_j, return_coord=False)    # (E, )
        ind_ij = torch.sigmoid((dist_ivj - self.c) / self.tau)  # (E, )
        vol_G = weight.sum()
        loss1 = -torch.log2(vol_G) / vol_G * (ind_ij * weight).sum()
        
        ind_sum_i = scatter_sum(ind_ij * weight, index=edge_index[0])    # (N, )
        embedding = self.embeddings(torch.arange(self.num_nodes).to(device))
        embedding = self.normalize(embedding)
        dist_pairs = hyp_lca(embedding[None], embedding[:, None, :], return_coord=False)
        ind_pairs = torch.sigmoid((dist_pairs - self.c) / self.tau)
        log_sum_dk = torch.log2(torch.sum(ind_pairs * degrees, -1))   # (N, )
        loss2 = 1 / vol_G * torch.sum(log_sum_dk * ind_sum_i)
        
        loss3 = -1 / vol_G * torch.sum(degrees * torch.log2(degrees)) + torch.log2(vol_G)
        return loss1 + loss2 + loss3