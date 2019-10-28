from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import hashlib
import logging
import math
import matplotlib
matplotlib.use('agg')
import os
import pickle
import random
import sys
import warnings

from absl import flags
import graph_nets as gn
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from grevnet_synthetic_data import *
from gnn import *
from graph_data import *
from loss import *
from utils import *

warnings.filterwarnings("ignore")

# Dataset params.
flags.DEFINE_string('dataset', '', '')
flags.DEFINE_string('ckpt_dir', '', '')
flags.DEFINE_integer('number_to_generate', 50, '')
flags.DEFINE_string('output_dir', '', '')
flags.DEFINE_integer('node_embedding_dim', 100, '')

BATCH_SIZE = 32
FLAGS = tf.app.flags.FLAGS

dataset = GraphDataset(FLAGS.dataset, FLAGS.node_embedding_dim)
n_nodes_distro = dataset.train_n_nodes()
sess = reset_sess()
latest_checkpoint = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
saver = tf.train.import_meta_graph("{}.meta".format(latest_checkpoint))
saver.restore(sess, latest_checkpoint)

values_map = {
    'sample_pred_adj': tf.get_collection('sample_pred_adj')[0],
    'sample_log_prob': tf.get_collection('sample_log_prob')[0],
    'sample_n_node': tf.get_collection('sample_n_node')[0],
}
graphs = []
iteration = 0
graph_ind = 0
n_nodes_ind = 0
while iteration < FLAGS.number_to_generate:
    print("iteration {}".format(iteration))
    n_node = []
    for _ in range(BATCH_SIZE):
        n_node.append(n_nodes_distro[n_nodes_ind])
    print("n node is {}".format(n_node))
    feed_dict = {'sample_n_node_placeholder:0': n_node}
    values = sess.run(values_map, feed_dict=feed_dict)
    pred_adj = values["sample_pred_adj"]
    log_probs = values["sample_log_prob"]
    print("tot num nodes {}".format(np.sum(n_node)))
    adjacency = np.where(pred_adj > 0.5, np.ones_like(pred_adj),
                         np.zeros_like(pred_adj))
    n_node_cum = np.cumsum(n_node)
    start_ind = 0
    for i in range(BATCH_SIZE):
        end_ind = n_node_cum[i]
        graph = adjacency[start_ind:end_ind, start_ind:end_ind]
        graph = nx.convert_matrix.from_numpy_matrix(graph)
        log_prob_graph = np.sum(log_probs[start_ind:end_ind]) / (end_ind - start_ind)
        graphs.append(graph)
        start_ind = end_ind
        print("ind is {} log prob is {}".format(i, log_prob_graph))
        visualize_graph(graph, os.path.join(FLAGS.output_dir, "graph_{}.png".format(graph_ind)))
        graph_ind += 1
    iteration += 1
    n_nodes_ind += 1
pickle.dump(graphs, open(os.path.join(FLAGS.output_dir, 'graphs.p'), 'wb'))
