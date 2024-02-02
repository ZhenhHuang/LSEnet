import torch
import networkx as nx
import torch_geometric.datasets
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import Planetoid, Amazon, StochasticBlockModelDataset
from torch_geometric.utils import from_networkx, negative_sampling
from torch_scatter import scatter_sum
from utils.utils import index2adjacency, adjacency2index
import urllib.request
import io
import zipfile
import numpy as np


def load_data(configs):
    dataset = None
    if configs.dataset in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo']:
        dataset = PygDataset(root=configs.root_path, name=configs.dataset)
    elif configs.dataset == 'KarateClub':
        dataset = KarateClub()
    elif configs.dataset == 'FootBall':
        dataset = Football()
    elif configs.dataset in ['eat', 'bat', 'uat']:
        dataset = ATsDataset(root=configs.root_path, name=configs.dataset)
    elif configs.dataset == 'SBM':
        dataset = SBMDataset(root=configs.root_path)
    data = {}
    data['feature'] = dataset.feature
    data['num_features'] = dataset.num_features
    data['edge_index'] = dataset.edge_index
    data['degrees'] = dataset.degrees
    data['weight'] = dataset.weight
    data['num_nodes'] = dataset.num_nodes
    data['labels'] = dataset.labels
    data['num_classes'] = dataset.num_classes
    data['neg_edge_index'] = dataset.neg_edge_index
    data['adj'] = dataset.adj
    return data


def mask_edges(edge_index, neg_edges, val_prop=0.05, test_prop=0.1):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:, n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, val_edges_neg, test_edges_neg], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)


class KarateClub:
    def __init__(self):
        data = torch_geometric.datasets.KarateClub()
        self.feature = data.x
        self.num_features = data.x.shape[1]
        self.num_nodes = data.x.shape[0]
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.y.tolist()
        self.num_classes = len(np.unique(self.labels))
        self.neg_edge_index = negative_sampling(data.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)


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
        self.feature = torch.eye(data.num_nodes)
        self.num_features = data.num_nodes
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.value.tolist()
        self.num_classes = len(np.unique(self.labels))
        self.neg_edge_index = negative_sampling(data.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)


class PygDataset:
    def __init__(self, root, name='Cora'):
        if name in ['Cora', 'Citeseer', 'Pubmed']:
            dataset = Planetoid(root, name)
        else:
            dataset = Amazon(root, name)
        data = dataset.data
        self.num_nodes = data.x.shape[0]
        self.feature = data.x
        self.num_features = data.x.shape[1]
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.y.tolist()
        self.num_classes = len(np.unique(self.labels))
        self.neg_edge_index = negative_sampling(data.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)


class ATsDataset:
    def __init__(self, root, name='eat'):
        adj = np.load(f'{root}/{name}/{name}_adj.npy')
        feat = np.load(f'{root}/{name}/{name}_feat.npy')
        label = np.load(f'{root}/{name}/{name}_label.npy')
        self.num_nodes = feat.shape[0]
        self.feature = torch.tensor(feat).float()
        self.num_features = feat.shape[1]
        self.edge_index = adjacency2index(torch.tensor(adj))
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = list(label)
        self.num_classes = len(np.unique(self.labels))
        self.neg_edge_index = negative_sampling(self.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)


class SBMDataset:
    def __init__(self, root, num_classes=5, num_nodes=200, p_in=0.6, p_out=0.03):
        num_classes = num_classes
        p = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    p[i, j] = p_in
                else:
                    p[i, j] = p_out

        data = StochasticBlockModelDataset(root,
                                              [num_nodes / num_classes] * num_classes,
                                              p, num_nodes=num_nodes)[0]
        data.x = torch.eye(data.num_nodes)
        self.num_nodes = data.x.shape[0]
        self.feature = data.x
        self.num_features = data.x.shape[1]
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        self.labels = data.y.tolist()
        self.num_classes = len(np.unique(self.labels))
        self.neg_edge_index = negative_sampling(data.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)


