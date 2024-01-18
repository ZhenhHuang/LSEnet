import torch
from utils.lca import hyp_lca, equiv_weights
from utils.utils import Frechet_mean_poincare
import networkx as nx
import numpy as np
from copy import deepcopy


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor, coords=None, tree_index=None, is_leaf=False):
        self.index = index  # T_alpha
        self.embeddings = embeddings    # coordinates of nodes in T_alpha
        self.children = []
        self.coords = coords  # node coordinates
        self.tree_index = tree_index
        self.is_leaf = is_leaf


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


def construct_tree(L_nodes: torch.LongTensor, manifold, embeddings: torch.Tensor, L_weights: list, height, k, nodes_count):
    root = Node(L_nodes, embeddings[L_nodes], tree_index=nodes_count)
    root.coords = Frechet_mean_poincare(manifold, root.embeddings)
    if k == height:
        for i in L_nodes:
            root.children.append(Node([i], embeddings[i], embeddings[i], tree_index=i.item(), is_leaf=True))
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
            root.children.append(Node([i], embeddings[i], embeddings[i], tree_index=i.item(), is_leaf=True))
        return root
    for child in children:
        nodes_count += 1
        root.children.append(construct_tree(torch.tensor(child).long(),
                                            manifold, embeddings, L_weights, height, k + 1, nodes_count))
    return root


def to_networkx_tree(tree: Node, embeddings):
    edges = []
    nodes = []

    def search_edges(node: Node, nodes_list, edges_list, height=0):
        # if node is a leaf
        if len(node.children) < 1:
            nodes_list.append(
                (
                    node.tree_index,
                 {'coords': node.coords.reshape(-1),
                  'is_leaf': node.is_leaf,
                  'children': node.index,
                  'height': height}
                 )
            )
            return
        for child in node.children:
            edges_list.append((node.tree_index, child.tree_index))
            search_edges(child, nodes_list, edges_list, height + 1)
        nodes_list.append(
            (
                node.tree_index,
                {'coords': node.coords.reshape(-1),
                 'is_leaf': node.is_leaf,
                 'children': node.index,
                 'height': height}
            )
        )

    search_edges(tree, nodes, edges, height=0)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph