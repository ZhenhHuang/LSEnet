import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import from_networkx
from torch_scatter import scatter_sum
import urllib.request
import io
import zipfile
import numpy as np


def load_data(configs):
    # dataset = KarateClub()
    dataset = Football()
    # dataset = Cora()
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


class Football:
    def __init__(self):
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        graph = nx.parse_gml(gml)  # parse gml data

        data = from_networkx(graph)
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.value.tolist()


class Cora:
    def __init__(self):
        dataset = Planetoid('D:\datasets\Graphs', 'cora')
        data = dataset.data
        self.num_nodes = data.x.shape[0]
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.y.tolist()