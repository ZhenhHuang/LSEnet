import torch
from utils.lca import hyp_lca, equiv_weights
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


def construct_tree(nodes_list: torch.LongTensor, manifold, node_embeddings: torch.Tensor,
                   coords_list: dict, ass_list: dict, height, num_nodes):
    nodes_count = num_nodes

    def _plan_DFS(L_nodes: torch.LongTensor, _manifold, embeddings: torch.Tensor, L_ass: dict, _height, k):
        nonlocal nodes_count
        root = Node(L_nodes, embeddings[L_nodes], tree_index=nodes_count)
        root.coords = Frechet_mean_poincare(_manifold, root.embeddings)

        if len(L_nodes) == 0:
            return None

        if k > _height or len(L_nodes) == 1:
            return Node([L_nodes[0]], embeddings[L_nodes[0]], embeddings[L_nodes[0]],
                        tree_index=L_nodes[0].item(), is_leaf=True)

        if k == _height:
            for i in L_nodes:
                root.children.append(Node([i], embeddings[i], embeddings[i], tree_index=i.item(), is_leaf=True))
            return root

        temp_ass = L_ass[k][L_nodes].cpu()
        children = []
        for j in range(temp_ass.shape[-1]):
            temp_child = L_nodes[temp_ass[:, j].nonzero().flatten()]
            if len(temp_child) > 0:
                children.append(temp_child)

        if len(children) <= 1:
            for i in L_nodes:
                root.children.append(Node([i], embeddings[i], embeddings[i], tree_index=i.item(), is_leaf=True))
            return root
        for child in children:
            nodes_count += 1
            child_node = _plan_DFS(child, _manifold, embeddings, L_ass, _height, k + 1)
            if child_node is not None:
                root.children.append(child_node)
        return root

    def _plan_BFS(L_nodes: torch.LongTensor, _manifold, embeddings: torch.Tensor,
                  L_coords: dict, L_ass: dict, _height):
        nonlocal nodes_count
        que = Queue()
        root = Node(L_nodes, embeddings[L_nodes], coords=L_coords[0].cpu(), tree_index=nodes_count, height=0)
        que.put(root)

        while not que.empty():
            node = que.get()
            L_nodes = node.index
            k = node.height + 1
            if k == height:
                for i in L_nodes:
                    node.children.append(Node([i], embeddings[i], embeddings[i],
                                              tree_index=i.item(), is_leaf=True, height=k))
            else:
                temp_ass = L_ass[k][L_nodes].cpu()
                for j in range(temp_ass.shape[-1]):
                    temp_child = L_nodes[temp_ass[:, j].nonzero().flatten()]
                    if len(temp_child) > 0:
                        nodes_count += 1
                        child_node = Node(temp_child, embeddings[temp_child], coords=L_coords[k][j].cpu(),
                                          tree_index=nodes_count, height=k)
                        node.children.append(child_node)
                        que.put(child_node)
        return root

    # return _plan_DFS(nodes_list, manifold, node_embeddings, ass_list, height, 1)
    return _plan_BFS(nodes_list, manifold, node_embeddings, coords_list, ass_list, height)


def to_networkx_tree(tree: Node, manifold):
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
            edges_list.append(
                (node.tree_index,
                 child.tree_index,
                 {'weight': torch.sigmoid(1.-manifold.dist_cpu(node.coords, child.coords)).item()}
                 ))
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
