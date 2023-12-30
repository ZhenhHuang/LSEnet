import torch
from lca import hyp_lca
import networkx as nx
import numpy as np
from copy import deepcopy


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor):
        self.index = index
        self.embeddings = embeddings
        self.children = []


def dfs(v: int, L_node: list, comp, used, adj):
    used[v] = 1
    comp.append(L_node[v])
    for u in torch.where(adj[v] == 1)[0]:
        if used[u] == 0:
            dfs(u, L_node, comp, used, adj)


def DFS_Comps(L_nodes: list, I) -> list[list]:
    used = [0] * len(L_nodes)
    results = []
    for i in range(len(L_nodes)):
        comp = []
        if used[i] == 0:
            dfs(i, L_nodes, comp, used, I)
        if len(comp) >= 1:
            results.append(comp)
    return results


def I_ij_k(L_nodes, embedding, height, k, c=0.5, epsilon=0.99, tau=0.05) -> torch.Tensor:
    if k == height:
        connect = torch.eye(len(L_nodes))
    else:
        dist_pairs = hyp_lca(embedding[None], embedding[:, None, :], return_coord=False, proj_hyp=False)
        radius_k = k * c
        dist_pairs_inversion = radius_k ** 2 / dist_pairs
        ratio = (radius_k - dist_pairs_inversion) / radius_k
        ind_pairs = torch.sigmoid((ratio - 0.5 / k) / tau)
        connect = (ind_pairs > epsilon).long()
    i, j = torch.where(connect == 1)
    edges = []
    for (ii, jj) in zip(i, j):
        edges.append((L_nodes[ii], L_nodes[jj]))
    return edges


def construct_tree(L_nodes: list, embeddings: torch.Tensor, K, c, k=1):
    root = Node(L_nodes, embeddings[torch.tensor(L_nodes).long()])
    if k == K:
        for i in L_nodes:
            root.children.append(Node([i], embeddings[i]))
        return root
    if k > K or len(L_nodes) <= 1:
        return root
    edges = I_ij_k(L_nodes, root.embeddings, K, k=k, c=c)
    G = nx.Graph()
    G.add_nodes_from(L_nodes)
    G.add_edges_from(edges)
    # children = DFS_Comps(L_nodes, I)
    children = nx.connected_components(G)
    children = [list(child) for child in children]
    if len(children) <= 1:
        for i in L_nodes:
            root.children.append(Node([i], embeddings[i]))
        return root
    for child in children:
        root.children.append(construct_tree(child, embeddings, K, c, k + 1))
    return root
