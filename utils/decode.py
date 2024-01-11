import torch
from utils.lca import hyp_lca, equiv_weights
from utils.utils import Frechet_mean
import networkx as nx
import numpy as np
from copy import deepcopy


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor, coords=None):
        self.index = index  # T_alpha
        self.embeddings = embeddings    # coordinates of nodes in T_alpha
        self.children = []
        self.coords = coords  # node coordinates


def build_equiv_graph(L_nodes, embedding, weights, height, k, epsilon=0.9) -> torch.Tensor:
    if k == height:
        connect = torch.eye(L_nodes.shape[0])
    else:
        ind_pairs = weights[L_nodes[:, None], L_nodes[None]]
        connect = (ind_pairs > epsilon).long()
    i, j = torch.where(connect == 1)
    edges = torch.stack([L_nodes[i], L_nodes[j]], dim=-1)
    edges = [tuple(e.tolist()) for e in edges]
    return edges


def construct_tree(L_nodes: torch.LongTensor, manifold, embeddings: torch.Tensor, L_weights: list, height, k=1):
    root = Node(L_nodes, embeddings[L_nodes])
    root.coords = Frechet_mean(manifold, root.embeddings)
    if k == height:
        for i in L_nodes:
            root.children.append(Node([i], embeddings[i], embeddings[i]))
        return root
    if k > height or len(L_nodes) <= 1:
        return root
    edges = build_equiv_graph(L_nodes, root.embeddings, L_weights[k].detach().cpu(), height, k=k)
    G = nx.Graph()
    G.add_nodes_from(L_nodes.tolist())
    G.add_edges_from(edges)
    children = nx.connected_components(G)
    children = [list(child) for child in children]
    if len(children) <= 1:
        for i in L_nodes:
            root.children.append(Node([i], embeddings[i], embeddings[i]))
        return root
    for child in children:
        root.children.append(construct_tree(torch.tensor(child).long(),
                                            manifold, embeddings, L_weights, height, k + 1))
    return root