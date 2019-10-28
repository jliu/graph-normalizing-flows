from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from itertools import product, permutations
import pickle
import random

from sklearn.manifold import spectral_embedding
from sklearn import datasets
import graph_nets as gn
import networkx as nx
import numpy as np
import tensorflow as tf

from utils import *

FEATURES = gn.utils_np.GRAPH_NX_FEATURES_KEY
DICT_IND = 1


# Gaussian noise features.
def add_gaussian_noise_features(graph, num_components=5, scale=1.0):
    for node in graph.nodes(data=True):
        node[DICT_IND][FEATURES] = np.random.normal(
            scale=scale, size=num_components).astype(np.float32)


# Convert nx grid representation to use a single number instead of a positional
# tuple to index nodes. Set edge and global features to be 0. For every node,
# add an edge to itself.
def convert_nx_repr(graph, add_node_features_fn):
    """Convert nx grid representation to use a single number instead of a
  positional tuple to index nodes. Add node features and empty edge features.
  """
    new_graph = nx.DiGraph(features=0)
    index_map = {}
    new_ind = 0
    for node in graph.nodes(data=True):
        index_map[node[0]] = new_ind
        new_graph.add_node(new_ind)
        new_graph.add_edge(new_ind, new_ind, features=0)
        new_ind += 1

    for edge in graph.edges(data=True):
        new_graph.add_edge(index_map[edge[0]], index_map[edge[1]], features=0)

    add_node_features_fn(new_graph)
    return new_graph


def preprocess_networkx_graphs(graphs, add_node_features_fn):
    to_ret = []
    for g in graphs:
        convert_graph = convert_nx_repr(g, add_node_features_fn)
        to_ret.append(convert_graph)
    return to_ret


class GraphDataset():
    def __init__(self, dataset_name, node_embedding_dim, gaussian_scale=1.0):
        filename = FILENAME_MAP[dataset_name]
        self.graphs = pickle.load(open(filename, 'rb'))
        self.add_node_features_fn = partial(add_gaussian_noise_features,
                                            num_components=node_embedding_dim,
                                            scale=gaussian_scale)
        self.train_graphs, self.test_graphs = self.process_and_split_graphs(
            self.graphs, node_embedding_dim, gaussian_scale,
            self.add_node_features_fn)
        self.train_index = 0
        self.test_index = 0

    def process_and_split_graphs(self, graphs, node_embedding_dim,
                                 gaussian_scale, add_node_features_fn):
        graphs_len = len(graphs)
        test_graphs = graphs[int(0.8 * graphs_len):]
        train_graphs = graphs[0:int(0.8 * graphs_len)]

        # Need to call to_directed so the message passing works correctly.
        test_graphs = [g.to_directed() for g in test_graphs]
        test_graphs = preprocess_networkx_graphs(test_graphs,
                                                 add_node_features_fn)
        train_graphs = [g.to_directed() for g in train_graphs]
        train_graphs = preprocess_networkx_graphs(train_graphs,
                                                  add_node_features_fn)
        return train_graphs, test_graphs

    def full_n_nodes(self):
        return [g.number_of_nodes() for g in self.graphs]

    def train_n_nodes(self):
        return [g.number_of_nodes() for g in self.train_graphs]

    def test_n_nodes(self):
        return [g.number_of_nodes() for g in self.test_graphs]

    def get_next_test_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.test_index == 0:
                random.shuffle(self.test_set)
            batch.append(self.test_graphs[self.test_index])
            self.index = (self.test_index + 1) % len(self.test_graphs)
        for g in batch:
            self.add_node_features_fn(g)
        return gn.utils_np.networkxs_to_graphs_tuple(batch)

    def get_random_test_batch(self, batch_size):
        return gn.utils_np.networkxs_to_graphs_tuple(
            random.choices(self.test_graphs, k=batch_size))

    def get_next_train_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.train_index == 0:
                random.shuffle(self.train_graphs)
            batch.append(self.train_graphs[self.train_index])
            self.index = (self.train_index + 1) % len(self.train_graphs)
        for g in batch:
            self.add_node_features_fn(g)
        return gn.utils_np.networkxs_to_graphs_tuple(batch)


class OverfitGraphDataset():
    def __init__(self,
                 dataset_name,
                 num_graphs,
                 train_batch_size,
                 node_embedding_dim,
                 graph_sizes=None,
                 gaussian_scale=1.0):
        filename = FILENAME_MAP[dataset_name]
        with open(filename, 'rb') as f:
            self.graphs = pickle.load(f)
        self.add_node_features_fn = partial(add_gaussian_noise_features,
                                            num_components=node_embedding_dim,
                                            scale=gaussian_scale)
        self.train_graphs, self.test_graphs = self.process_and_split_graphs(
            self.graphs, node_embedding_dim, gaussian_scale,
            self.add_node_features_fn)
        self.train_graphs = self.subset_graphs(self.train_graphs, num_graphs,
                                               train_batch_size, graph_sizes)
        self.train_index = 0
        self.test_index = 0

    def full_n_nodes(self):
        return [g.number_of_nodes() for g in self.train_graphs]

    def train_n_nodes(self):
        return [g.number_of_nodes() for g in self.train_graphs]

    def test_n_nodes(self):
        return [g.number_of_nodes() for g in self.train_graphs]


    def subset_graphs(self, graphs, num_graphs, train_batch_size, graph_sizes=None):
        graphs.sort(key=lambda x: x.number_of_nodes())
        if graph_sizes:
            subset = []
            size_map = {}
            for g in graphs:
                if g.number_of_nodes() in size_map:
                    size_map[g.number_of_nodes()].append(g)
                else:
                    size_map[g.number_of_nodes()] = [g]
            for s in graph_sizes:
                subset.append(size_map[s][0])
        else:
            subset = graphs[0:num_graphs]
        new_graphs = []
        ind = 0
        while len(new_graphs) < max(num_graphs, train_batch_size):
            new_graphs.append(
                subset[ind % len(subset)].copy(as_view=False))
            ind += 1
        return new_graphs

    def process_and_split_graphs(self, graphs, node_embedding_dim,
                                 gaussian_scale, add_node_features_fn):
        graphs_len = len(graphs)
        test_graphs = graphs[int(0.8 * graphs_len):]
        train_graphs = graphs[0:int(0.8 * graphs_len)]

        # Need to call to_directed so the message passing works correctly.
        test_graphs = [g.to_directed() for g in test_graphs]
        test_graphs = preprocess_networkx_graphs(test_graphs,
                                                 add_node_features_fn)
        train_graphs = [g.to_directed() for g in train_graphs]
        train_graphs = preprocess_networkx_graphs(train_graphs,
                                                  add_node_features_fn)
        return train_graphs, test_graphs

    def get_next_train_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.train_index == 0:
                random.shuffle(self.train_graphs)
            batch.append(self.train_graphs[self.train_index])
            self.index = (self.train_index + 1) % len(self.train_graphs)
        for g in batch:
            self.add_node_features_fn(g)
        return gn.utils_np.networkxs_to_graphs_tuple(batch)


class NoisyGraphDataset():
    def __init__(self,
                 dataset_name,
                 node_embedding_dim,
                 deletion_prob=0.5,
                 gaussian_scale=1.0):
        filename = FILENAME_MAP[dataset_name]
        self.graphs = pickle.load(open(filename, 'rb'))
        self.add_node_features_fn = partial(add_gaussian_noise_features,
                                            num_components=node_embedding_dim,
                                            scale=gaussian_scale)
        self.train_graphs, self.test_graphs = self.process_and_split_graphs(
            self.graphs, node_embedding_dim, gaussian_scale,
            self.add_node_features_fn)
        self.train_index = 0
        self.test_index = 0
        self.deletion_prob = deletion_prob

    def process_and_split_graphs(self, graphs, node_embedding_dim,
                                 gaussian_scale, add_node_features_fn):
        graphs_len = len(graphs)
        test_graphs = graphs[int(0.8 * graphs_len):]
        train_graphs = graphs[0:int(0.8 * graphs_len)]

        # Need to call to_directed so the message passing works correctly.
        test_graphs = preprocess_networkx_graphs(test_graphs,
                                                 add_node_features_fn)
        train_graphs = preprocess_networkx_graphs(train_graphs,
                                                  add_node_features_fn)
        return train_graphs, test_graphs


    def perturb_batch(self, batch):
        new_batch = []
        num_removed = []
        for g in batch:
            to_remove = []
            new_g = g.copy(as_view=False)
            for e in new_g.edges():
                if random.random() < self.deletion_prob:
                    to_remove.append(e)
            #new_g.remove_edges_from(to_remove)
            num_removed.append(len(to_remove))
            new_batch.append(new_g)
        return new_batch, num_removed

    def get_next_test_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.test_index == 0:
                random.shuffle(self.test_set)
            batch.append(self.test_graphs[self.test_index])
            self.index = (self.test_index + 1) % len(self.test_graphs)
        for g in batch:
            self.add_node_features_fn(g)
        return gn.utils_np.networkxs_to_graphs_tuple(batch)

    def get_random_test_batch(self, batch_size):
        return gn.utils_np.networkxs_to_graphs_tuple(
            random.choices(self.test_graphs, k=batch_size))

    def get_next_train_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.train_index == 0:
                random.shuffle(self.train_graphs)
            batch.append(self.train_graphs[self.train_index])
            self.index = (self.train_index + 1) % len(self.train_graphs)
        for g in batch:
            self.add_node_features_fn(g)
        noisy_batch, num_removed = self.perturb_batch(batch)
        noisy_batch = [g.to_directed() for g in noisy_batch]
        batch = [g.to_directed() for g in batch]
        return gn.utils_np.networkxs_to_graphs_tuple(
            batch), gn.utils_np.networkxs_to_graphs_tuple(batch), num_removed


FILENAME_MAP = {
    'graph_rnn_grid':
    'training_graphs/GraphRNN_RNN_grid_4_128_train_0.dat',
    'graph_rnn_protein':
    'training_graphs/GraphRNN_RNN_protein_4_128_train_0.dat',
    'graph_rnn_ego':
    'training_graphs/GraphRNN_RNN_citeseer_4_128_train_0.dat',
    'graph_rnn_community':
    'training_graphs/GraphRNN_RNN_caveman_4_128_train_0.dat',
    'graph_rnn_ego_small':
    'training_graphs/GraphRNN_RNN_citeseer_small_4_64_train_0.dat',
    'graph_rnn_community_small':
    'training_graphs/GraphRNN_RNN_caveman_small_4_64_train_0.dat',
    'graph_rnn_community_medium':
    'training_graphs/GraphRNN_RNN_community_medium_4_128_train_0.dat',
    'graph_rnn_ego_medium':
    'training_graphs/GraphRNN_RNN_ego_medium_4_128_train_0.dat',
}
