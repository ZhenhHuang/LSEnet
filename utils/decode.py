import torch
from utils.utils import Frechet_mean_poincare
import networkx as nx
import numpy as np
from copy import deepcopy
from queue import Queue


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor, coords=None,
                 tree_index=None, is_leaf=False, height: int = None):
        self.index = index  # T_alpha
        self.embeddings = embeddings  # coordinates of nodes in T_alpha
        self.children = []
        self.coords = coords  # node coordinates
        self.tree_index = tree_index
        self.is_leaf = is_leaf
        self.height = height


def construct_tree(nodes_list: torch.LongTensor, manifold, coords_list: dict,
                   ass_list: dict, height, num_nodes):
    nodes_count = num_nodes
    que = Queue()
    root = Node(nodes_list, coords_list[height][nodes_list].cpu(),
                coords=coords_list[0].cpu(), tree_index=nodes_count, height=0)
    que.put(root)

    while not que.empty():
        node = que.get()
        L_nodes = node.index
        k = node.height + 1
        if k == height:
            for i in L_nodes:
                node.children.append(Node(i.reshape(-1), coords_list[height][i].cpu(), coords=coords_list[k][i].cpu(),
                                          tree_index=i.item(), is_leaf=True, height=k))
        else:
            temp_ass = ass_list[k][L_nodes].cpu()
            for j in range(temp_ass.shape[-1]):
                temp_child = L_nodes[temp_ass[:, j].nonzero().flatten()]
                if len(temp_child) > 0:
                    nodes_count += 1
                    child_node = Node(temp_child, coords_list[height][temp_child].cpu(),
                                      coords=coords_list[k][j].cpu(),
                                      tree_index=nodes_count, height=k)
                    node.children.append(child_node)
                    que.put(child_node)
    return root


def to_networkx_tree(root: Node, manifold, height):
    edges_list = []
    nodes_list = []
    que = Queue()
    que.put(root)
    nodes_list.append(
        (
            root.tree_index,
            {'coords': root.coords.reshape(-1),
             'is_leaf': root.is_leaf,
             'children': root.index,
             'height': root.height}
        )
    )

    while not que.empty():
        cur_node = que.get()
        if cur_node.height == height:
            break
        for node in cur_node.children:
            nodes_list.append(
                (
                    node.tree_index,
                    {'coords': node.coords.reshape(-1),
                     'is_leaf': node.is_leaf,
                     'children': node.index,
                     'height': node.height}
                )
            )
            edges_list.append(
                (
                    cur_node.tree_index,
                    node.tree_index,
                    {'weight': torch.sigmoid(1. - manifold.dist(cur_node.coords, node.coords)).item()}
                )
            )
            que.put(node)

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)
    return graph
