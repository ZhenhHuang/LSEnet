import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx
from torch_scatter import scatter_sum


def load_data(configs):
    dataset = KarateClub()
    data = {}
    data['edge_index'] = dataset.edge_index
    data['degrees'] = dataset.degrees
    data['weight'] = dataset.weight
    data['num_nodes'] = dataset.num_nodes
    data['labels'] = dataset.labels
    return data


class KarateClub:
    def __init__(self):
        graph = nx.karate_club_graph()
        data = from_networkx(graph)
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.weight = data.weight
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.club