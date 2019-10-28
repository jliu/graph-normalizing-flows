from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import os
import pickle
import warnings

from absl import flags
import tensorflow as tf

from graph_data import *
from utils import *

warnings.filterwarnings("ignore")

flags.DEFINE_string('checkpoint', '', '')
flags.DEFINE_integer('random_seed', 12345, '')

# Input example params.
flags.DEFINE_integer('train_batch_size', 32, 'Dimension of node embeddings.')
flags.DEFINE_integer('num_examples', 200000, 'Dimension of node embeddings.')
flags.DEFINE_string('dataset', 'graph_rnn_ego', '')
flags.DEFINE_integer('node_embedding_dim', 100,
                     'Dimension of node embeddings.')
flags.DEFINE_string('output_file', '', '')
flags.DEFINE_integer('run_number', 0, '')
flags.DEFINE_float('gaussian_scale', 1.0,
                   'Scale to use for random Gaussian features.')
flags.DEFINE_bool('output_pickled', True, '')
flags.DEFINE_integer(
    'num_graphs', 0,
    'Number of graph types to randomly pick if only using a subset of training data.'
)
flags.DEFINE_string('graph_sizes', '50,100', '')
FLAGS = tf.app.flags.FLAGS

#os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.run_number)
# Logging and print options.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})

graph_sizes = [int(x) for x in FLAGS.graph_sizes.split(',')] if FLAGS.graph_sizes else None
if FLAGS.num_graphs > 0:
    dataset = OverfitGraphDataset(FLAGS.dataset, FLAGS.num_graphs, FLAGS.train_batch_size, FLAGS.node_embedding_dim, graph_sizes)
else:
    dataset = GraphDataset(FLAGS.dataset, FLAGS.node_embedding_dim)

sess = reset_sess()
saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.checkpoint))
saver.restore(sess, FLAGS.checkpoint)

values_map = {
    'gnn_output': tf.get_collection('gnn_output')[0],
    'total_incorrect_edges': tf.get_collection('total_incorrect_edges')[0],
}

batch_num = 0
num_examples = 0
file_number = 0
filename_prefix = os.environ.get('MLPATH')
filename_template_str = "{}_{}_{{}}.{}"
filename_template = filename_template_str.format(
    FLAGS.output_file, FLAGS.run_number,
    "p") if FLAGS.output_pickled else filename_template_str.format(
        FLAGS.output_file, FLAGS.run_number, "tfrecord")
filename_template = os.path.join(filename_prefix, filename_template)
filename = filename_template.format(file_number)
if not FLAGS.output_pickled:
    writer = tf.io.TFRecordWriter(filename)
else:
    open(filename, 'a').close()
total_n_node = 0
node_embeddings = np.empty([0, FLAGS.node_embedding_dim])
n_node = np.empty([0], dtype=np.int32)

# TODO(jyliu): change this
while True:
    generated_enough_examples = num_examples > FLAGS.num_examples
    filename_maxed_out = total_n_node * FLAGS.node_embedding_dim * 4 > 100e6 if FLAGS.output_pickled else os.path.getsize(
        filename) > 100e6
    if generated_enough_examples and filename_maxed_out:
        break
    if not FLAGS.output_pickled and os.path.getsize(filename) > 100e6:
        writer.close()
        file_number += 1
        filename = filename_template.format(file_number)
        if not FLAGS.output_pickled:
            writer = tf.io.TFRecordWriter(filename)
    elif FLAGS.output_pickled and total_n_node * FLAGS.node_embedding_dim * 4 > 100e6:
        with open(filename, 'wb') as f:
            pickle.dump((node_embeddings, n_node), f)
        file_number += 1
        filename = filename_template.format(file_number)
        total_n_node = 0
        node_embeddings = np.empty([0, FLAGS.node_embedding_dim])
        n_node = np.empty([0], dtype=np.int32)
    graphs_tuple = dataset.get_next_train_batch(FLAGS.train_batch_size)
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
    cs = np.cumsum(graphs_tuple.n_node)
    start_ind = 0
    end_ind = 0
    total_n_node += np.sum(graphs_tuple.n_node)

    if FLAGS.output_pickled:
        n_node = np.append(n_node, graphs_tuple.n_node, axis=0)
        node_embeddings = np.append(node_embeddings,
                                    values["gnn_output"],
                                    axis=0)
    else:
        for j in range(FLAGS.train_batch_size):
            end_ind = cs[j]
            node_features = values["gnn_output"][start_ind:end_ind]
            feature_dict = {
                'zs':
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=values["gnn_output"][start_ind:end_ind].flatten())),
            }
            example = tf.train.Example(features=tf.train.Features(
                feature=feature_dict))
            writer.write(example.SerializeToString())
            start_ind = end_ind
    num_examples += FLAGS.train_batch_size

    if batch_num % 100 == 0:
        print("batch num {}, processed {} examples".format(
            batch_num, num_examples))
        total_incorrect_edges = values['total_incorrect_edges']
        print("total incorrect edges {}".format(total_incorrect_edges))
        print("n node {}".format(graphs_tuple.n_node))
        print("node embeddings {}".format(graphs_tuple.nodes))
    batch_num += 1
if not FLAGS.output_pickled:
    writer.close()
