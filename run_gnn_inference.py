from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import os
import warnings

from absl import flags
import tensorflow as tf

from graph_data import *
from utils import *

warnings.filterwarnings("ignore")

flags.DEFINE_string('checkpoint', '', '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_integer('tf_random_seed', 601904901297, '')

# Input example params.
flags.DEFINE_integer('train_batch_size', 32, 'Dimension of node embeddings.')
flags.DEFINE_integer('num_train_iters', 10, 'Dimension of node embeddings.')
flags.DEFINE_string('dataset', 'graph_rnn_community', '')
flags.DEFINE_integer('node_embedding_dim', 40, 'Dimension of node embeddings.')
flags.DEFINE_string('node_features', 'gaussian',
                    'Can be laplacian, gaussian, or zero.')
flags.DEFINE_string('output_file', '', '')
flags.DEFINE_integer('run_number', 0, '')
flags.DEFINE_float('gaussian_scale', 1.0,
                   'Scale to use for random Gaussian features.')
FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.run_number)

# Logging and print options.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})
tf.random.set_random_seed(FLAGS.tf_random_seed)
random.seed(FLAGS.random_seed)

NODE_FEATURES_MAP = {
    'laplacian':
    partial(add_laplacian_features, num_components=FLAGS.node_embedding_dim),
    'gaussian':
    partial(add_gaussian_noise_features,
            num_components=FLAGS.node_embedding_dim,
            scale=FLAGS.gaussian_scale),
    'zeros':
    partial(add_zero_features, num_components=FLAGS.node_embedding_dim),
    'positional':
    partial(add_positional_encoding_features,
            num_components=FLAGS.node_embedding_dim),
}
add_node_features_fn = NODE_FEATURES_MAP[FLAGS.node_features]

DATASET_MAP = {
    'graph_rnn_grid':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_grid_4_128_train_0.dat',
            add_node_features_fn),
    'graph_rnn_protein':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_protein_4_128_train_0.dat',
            add_node_features_fn),
    'graph_rnn_ego':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_citeseer_4_128_train_0.dat',
            add_node_features_fn),
    'graph_rnn_community':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_caveman_4_128_train_0.dat',
            add_node_features_fn),
    'graph_rnn_ego_small':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_citeseer_small_4_64_train_0.dat',
            add_node_features_fn),
    'graph_rnn_community_small':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_caveman_small_4_64_train_0.dat',
            add_node_features_fn),
    'graph_rnn_community_test':
    partial(load_graph_rnn_dataset_test,
            'training_graphs/GraphRNN_RNN_caveman_4_128_train_0.dat',
            add_node_features_fn),
    'graph_rnn_community_medium':
    partial(load_grevnet_graph_rnn_dataset,
            'training_graphs/GraphRNN_RNN_community_medium_4_128_train_0.dat',
            add_node_features_fn),
}

dataset = DATASET_MAP[FLAGS.dataset]()

sess = reset_sess()
saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.checkpoint))
saver.restore(sess, FLAGS.checkpoint)

values_map = {
    'gnn_output': tf.get_collection('gnn_output')[0],
    'total_incorrect_edges': tf.get_collection('total_incorrect_edges')[0],
    'pred_adj': tf.get_collection('pred_adj')[0],
    'true_adj': tf.get_collection('true_adj')[0]
}

complete_graphs = []
pred_graphs = []

for i in range(FLAGS.num_train_iters):
    graphs_tuple = dataset.get_next_test_batch(FLAGS.train_batch_size)
    feed_dict = {}
    feed_dict["true_graph_phs/nodes:0"] = graphs_tuple.nodes
    feed_dict["true_graph_phs/edges:0"] = graphs_tuple.edges
    feed_dict["true_graph_phs/receivers:0"] = graphs_tuple.receivers
    feed_dict["true_graph_phs/senders:0"] = graphs_tuple.senders
    feed_dict["true_graph_phs/globals:0"] = graphs_tuple.globals
    feed_dict["true_graph_phs/n_node:0"] = graphs_tuple.n_node
    feed_dict["true_graph_phs/n_edge:0"] = graphs_tuple.n_edge
    feed_dict["is_training:0"] = False

    values = sess.run(values_map, feed_dict=feed_dict)
    print("iteration num {}".format(i))
    total_incorrect_edges = values['total_incorrect_edges']
    print("total_incorrect_edges {}".format(total_incorrect_edges))
    true_adj = values['true_adj']
    pred_adj = values['pred_adj']
    pred_adj_raw = np.array(pred_adj)
    pred_adj = np.where(pred_adj > 0.5, np.ones_like(pred_adj),
                        np.zeros_like(pred_adj))
    ind = np.cumsum(graphs_tuple.n_node)
    start = 0
    correct_edges = pred_adj * true_adj
    false_positives = pred_adj - correct_edges
    false_negatives = true_adj - correct_edges
    false_positive_scores = false_positives * pred_adj_raw
    false_positive_scores = false_positive_scores[np.nonzero(
        false_positive_scores)]
    if len(false_positive_scores) > 0:
        print("false positive quartiles: {}".format(
            np.quantile(false_positive_scores, [0, 0.25, 0.5, 0.75, 1.0])))
    false_negative_scores = false_negatives * pred_adj_raw
    false_negative_scores = false_negative_scores[np.nonzero(
        false_negative_scores)]
    if len(false_negative_scores) > 0:
        print("false negative quartiles: {}".format(
            np.quantile(false_negative_scores, [0, 0.25, 0.5, 0.75, 1.0])))

    for i in range(FLAGS.train_batch_size):
        end = ind[i]
        pred_mat = pred_adj[start:end, start:end]
        true_mat = true_adj[start:end, start:end]
        correct_edges_mat = pred_mat * true_mat
        false_positives_mat = pred_mat - correct_edges_mat
        false_negatives_mat = true_mat - correct_edges_mat
        complete_mat = correct_edges_mat + 2 * false_positives_mat + 3 * false_negatives_mat
        complete_graph = nx.from_numpy_matrix(complete_mat)
        complete_graphs.append(complete_graph)
        pred_graph = nx.from_numpy_matrix(pred_mat)
        pred_graphs.append(pred_graph)
        start = end

with open("{}_complete.p".format(FLAGS.output_file), 'wb') as f:
    pickle.dump(complete_graphs, f)
with open("{}_pred.p".format(FLAGS.output_file), 'wb') as f:
    pickle.dump(pred_graphs, f)
