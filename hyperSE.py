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
    def __init__(self, num_nodes, d_hyp=2, height=2, temperature=0.1, c=0.5,
                 init_size=1.0, min_size=1e-2, max_size=0.999):
        super(HyperSE, self).__init__()
        self.k = torch.tensor([-1.0])
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.embeddings = nn.Embedding(num_nodes, embedding_dim=d_hyp)
        self.scale = nn.Parameter(torch.tensor([init_size]), requires_grad=True)
        self.embeddings.weight.data = project(
            self.scale * (2 * torch.rand(num_nodes, d_hyp) - 1), 
            k=self.k, eps=MIN_NORM)
        self.c = 1 / height
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
            device: torch.Device
        """
        if weight is None:
            weight = torch.ones(edge_index.shape[-1])
        weight = weight.to(device)
        edge_index = edge_index.to(device)
        degrees = degrees.to(device)
        loss = 0

        embedding = self.embeddings(torch.arange(self.num_nodes).to(device))
        embedding = self.normalize(embedding)

        vol_G = weight.sum()
        dist_pairs = hyp_lca(embedding[None], embedding[:, None, :], return_coord=False)
        ind_pairs = [torch.ones(self.num_nodes, self.num_nodes).to(device)]
        for k in range(1, self.height):
            ind_pairs_k = torch.sigmoid((dist_pairs - k * self.c) / self.tau)
            ind_pairs.append(ind_pairs_k)
        ind_pairs.append(torch.eye(self.num_nodes).to(device))
        for k in range(1, self.height + 1):
            log_sum_dl_k = torch.log2(torch.sum(ind_pairs[k] * degrees, -1))  # (N, )
            log_sum_dl_k_1 = torch.log2(torch.sum(ind_pairs[k-1] * degrees, -1))  # (N, )
            ind_i_j = ind_pairs[k][edge_index[0], edge_index[1]]
            weight_sum = scatter_sum(ind_i_j * weight, index=edge_index[0])  # (N, )
            d_log_sum_k = (degrees - weight_sum) * log_sum_dl_k  # (N, )
            d_log_sum_k_1 = (degrees - weight_sum) * log_sum_dl_k_1     # (N, )
            loss += torch.sum(d_log_sum_k - d_log_sum_k_1)
        loss = -1 / vol_G * loss
        return loss