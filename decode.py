import torch
from lca import hyp_lca


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor):
        self.index = index
        self.embeddings = embeddings
        self.children = []


def dfs(v: int, L_node: list, comp, used, adj):
    used[v] = 1
    comp.append(L_node[v])
    for u in adj[v]:
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


def I_ij_k(embedding, k, c=0.5, epsilon=0.5, tau=0.1) -> torch.Tensor:
    dist_pairs = hyp_lca(embedding[None], embedding[:, None, :], return_coord=False)
    ind_pairs = torch.sigmoid((dist_pairs - k * c) / tau)
    return (ind_pairs > epsilon).long()


def construct_tree(L_nodes: list, embeddings: torch.Tensor, K, c, k=1):
    root = Node(L_nodes, embeddings[torch.tensor(L_nodes).long()])
    if k >= K or len(L_nodes) <= 1:
        return root
    I = I_ij_k(root.embeddings, k=k, c=c)
    children = DFS_Comps(L_nodes, I)
    for child in children:
        root.children.append(construct_tree(child, embeddings, K, c, k + 1))
    return root
