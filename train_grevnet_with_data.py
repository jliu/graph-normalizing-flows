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
import tensorflow.python.debug as tf_debug
tfd = tfp.distributions
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

from grevnet_synthetic_data import *
from grevnet import *
from gnn import *
from graph_data import *
from loss import *
from utils import *

warnings.filterwarnings("ignore")

flags.DEFINE_string('gpu', '0', '')
flags.DEFINE_string('grad_output_file', '', '')
flags.DEFINE_bool('debug_grads', False, '')

# Attention params.
flags.DEFINE_integer('attn_output_dim', 64, '')
flags.DEFINE_string('attn_type', 'dm_attn', '')
flags.DEFINE_integer('attn_kq_dim', 64, '')
flags.DEFINE_integer('attn_v_dim', 64, '')
flags.DEFINE_integer('attn_num_heads', 1, '')
flags.DEFINE_integer('attn_concat_heads_output_dim', 64, '')
flags.DEFINE_bool('attn_concat', True, '')
flags.DEFINE_bool('attn_residual', False, '')
flags.DEFINE_bool('attn_layer_norm', False, '')

# Dataset params.
flags.DEFINE_string('dataset', 'graph_rnn_grid', '')
flags.DEFINE_bool('overfit_dataset', False, '')
flags.DEFINE_integer('overfit_num_graphs', 1, '')
flags.DEFINE_string('overfit_size_map', '', '')

# Training params.
flags.DEFINE_integer('train_epochs', 6, '')
flags.DEFINE_integer('train_batch_size', 32, '')
flags.DEFINE_bool('variable_dataset', False, '')
flags.DEFINE_integer('max_nodes', 1500, '')
flags.DEFINE_string('train_data_dir', '', '')
flags.DEFINE_integer('write_graphs_every_n_steps', 1000, '')
flags.DEFINE_integer('write_graphs_min_iter', 1000, '')
flags.DEFINE_integer('sample_size', 32, '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_string('logdir', 'test_runs/test_grevnet_fixed_encoder',
                    'Where to write training files.')
flags.DEFINE_integer('num_train_iters', 200000, '')
flags.DEFINE_integer('log_every_n_steps', 1, '')
flags.DEFINE_integer('summary_every_n_steps', 25, '')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, '')
flags.DEFINE_integer('save_every_n_steps', 500, '')

# Optimizer params.
flags.DEFINE_string(
    'lr_type', 'constant',
    'Can be constant, fixed_decay, polynomial_decay, or schedule.')
flags.DEFINE_float('lr', 1e-04, 'Learning rate for optimizer.')
flags.DEFINE_integer('lr_fixed_decay_steps', 1000, '')
flags.DEFINE_float('lr_fixed_decay_rate', 0.99, '')
flags.DEFINE_bool('lr_fixed_decay_staircase', False, '')
flags.DEFINE_integer('lr_schedule_rampup', 1000, '')
flags.DEFINE_integer('lr_schedule_hold', 2000, '')
flags.DEFINE_float('l2_regularizer_weight', 0.0000001,
                   'Used to regularizer weights and biases of MLP.')
flags.DEFINE_float('adam_beta1', 0.9, '')
flags.DEFINE_float('adam_beta2', 0.999, '')
flags.DEFINE_float('adam_epsilon', 1e-08, '')
flags.DEFINE_bool('clip_gradient_by_value', False,
                  'Whether to use value-based gradient clipping.')
flags.DEFINE_float('clip_gradient_value_lower', -1.0,
                   'Lower threshold for valued-based gradient clipping.')
flags.DEFINE_float('clip_gradient_value_upper', 5.0,
                   'Upper threshold for value-based gradient clipping.')
flags.DEFINE_bool('clip_gradient_by_norm', False,
                  'Whether to use norm-based gradient clipping.')
flags.DEFINE_float('clip_gradient_norm', 10.0,
                   'Value for norm-based gradient clipping.')

# GRevNet params.
flags.DEFINE_bool('use_gnf', True, '')
flags.DEFINE_bool('use_efficient_backprop', True, '')
flags.DEFINE_integer('num_coupling_layers', 10, '')
flags.DEFINE_bool('weight_sharing', False, '')

# GNN params.
flags.DEFINE_bool('residual', False, '')
flags.DEFINE_bool('use_batch_norm', True, '')
flags.DEFINE_bool('use_layer_norm', False, '')
flags.DEFINE_integer('num_layers', 3, 'Num of layers of MLP used in GNN.')
flags.DEFINE_integer('latent_dim', 2048, 'Latent dim of MLP used in GNN.')
flags.DEFINE_float('bias_init_stddev', 0.3,
                   'Used for initializing bias weights in GNN.')

# Node feature params.
flags.DEFINE_integer('node_embedding_dim', 200,
                     'Dimension of node embeddings.')

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
logdir_prefix = os.environ.get('MLPATH')
if not logdir_prefix:
    logdir_prefix = '.'
LOGDIR = os.path.join(logdir_prefix, FLAGS.logdir)
os.makedirs(LOGDIR)
GRAPHS_LOGDIR = os.path.join(LOGDIR, "generated_graphs")
os.makedirs(GRAPHS_LOGDIR)
if FLAGS.debug_grads:
    grads_folder = os.path.join(LOGDIR, "grads")
    os.makedirs(grads_folder)

# Logging and print options.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})
handlers = [logging.StreamHandler(sys.stdout)]
handlers.append(logging.FileHandler(os.path.join(LOGDIR, 'OUTPUT_LOG')))
logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger("logger")

tf.random.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)


class GrevnetDatasetFixed():
    def __init__(self, train_data_dir, train_batch_size):
        self.files = os.listdir(train_data_dir) * FLAGS.train_epochs
        self.file_ind = 0
        self.prev_graph_ind = 0
        self.prev_node_embedding_ind = 0
        self.train_batch_size = train_batch_size
        self.train_data_dir = train_data_dir
        with open(os.path.join(train_data_dir, self.files[self.file_ind]),
                  'rb') as f:
            d = pickle.load(f)
            self.node_embeddings = d[0]
            self.n_node = d[1]
            self.n_node_cs = np.cumsum(self.n_node)

    def train_batch(self):
        new_ind = self.prev_graph_ind + self.train_batch_size
        if new_ind > len(self.n_node):
            self.file_ind += 1
            print("****" * 50)
            print("Reading next file")
            with open(
                    os.path.join(self.train_data_dir,
                                 self.files[self.file_ind]), 'rb') as f:

                d = pickle.load(f)
                self.node_embeddings = d[0]
                self.n_node = d[1]
                self.prev_graph_ind = 0
                self.prev_node_embedding_ind = 0
                self.n_node_cs = np.cumsum(self.n_node)
                new_ind = self.prev_graph_ind + self.train_batch_size
        node_embeddings = self.node_embeddings[self.prev_node_embedding_ind:
                                               self.n_node_cs[new_ind - 1]]
        n_node = self.n_node[self.prev_graph_ind:new_ind]
        self.prev_graph_ind = new_ind
        self.prev_node_embedding_ind = self.n_node_cs[new_ind - 1]
        return node_embeddings, n_node


class GrevnetDatasetVariable():
    def __init__(self, train_data_dir, max_nodes):
        self.files = os.listdir(train_data_dir)
        self.file_ind = 0
        self.graph_ind = 0
        self.prev_graph_ind = 0
        self.prev_node_embedding_ind = 0
        self.max_nodes = max_nodes
        self.train_data_dir = train_data_dir
        with open(os.path.join(self.train_data_dir, self.files[self.file_ind]),
                  'rb') as f:
            d = pickle.load(f)
            self.node_embeddings = d[0]
            self.n_node = d[1]
            self.n_node_cs = np.cumsum(self.n_node)

    def train_batch(self):
        total_nodes = 0
        while True:
            if self.graph_ind >= len(self.n_node):
                node_embeddings = self.node_embeddings[
                    self.prev_node_embedding_ind:self.n_node_cs[self.graph_ind
                                                                - 1]]
                n_node = self.n_node[self.prev_graph_ind:self.graph_ind]
                self.file_ind += 1
                self.prev_graph_ind = 0
                self.graph_ind = 0
                self.prev_node_embedding_ind = 0
                filename = os.path.join(self.train_data_dir,
                                        self.files[self.file_ind])
                print("****" * 50)
                print("Reading next file {}".format(filename))
                with open(filename, 'rb') as f:
                    d = pickle.load(f)
                    self.node_embeddings = d[0]
                    self.n_node = d[1]
                    self.n_node_cs = np.cumsum(self.n_node)
                return node_embeddings, n_node
            if total_nodes + self.n_node[self.graph_ind] < self.max_nodes:
                total_nodes += self.n_node[self.graph_ind]
                self.graph_ind += 1
            else:
                break
        node_embeddings = self.node_embeddings[self.prev_node_embedding_ind:
                                               self.n_node_cs[self.graph_ind -
                                                              1]]
        n_node = self.n_node[self.prev_graph_ind:self.graph_ind]
        self.prev_graph_ind = self.graph_ind
        self.prev_node_embedding_ind = self.n_node_cs[self.graph_ind - 1]
        return node_embeddings, n_node


def transform_example(n_node):
    globals = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals, receivers, senders, n_edge


sizes = [int(x) for x in FLAGS.overfit_size_map.split(",")
         ] if FLAGS.overfit_size_map else None
dataset = OverfitGraphDataset(
    FLAGS.dataset, FLAGS.overfit_num_graphs, FLAGS.train_batch_size,
    FLAGS.node_embedding_dim,
    sizes) if FLAGS.overfit_dataset else GraphDataset(FLAGS.dataset,
                                                      FLAGS.node_embedding_dim)
node_embeddings_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, FLAGS.node_embedding_dim],
    name='node_embeddings_placeholder')
n_node_placeholder = tf.placeholder(dtype=tf.int32,
                                    shape=[FLAGS.train_batch_size],
                                    name='n_node_placeholder')

# Define GNN and output.
edges, globals, receivers, senders, n_edge = transform_example(
    n_node_placeholder)
graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                     edges=edges,
                                     globals=globals,
                                     receivers=receivers,
                                     senders=senders,
                                     n_node=n_node_placeholder,
                                     n_edge=n_edge)
batch_n_node = tf.reduce_sum(n_node_placeholder)

make_mlp_fn = partial(make_mlp_model,
                      FLAGS.latent_dim,
                      FLAGS.node_embedding_dim / 2,
                      FLAGS.num_layers,
                      activation=tf.nn.relu,
                      l2_regularizer_weight=0.000001,
                      bias_init_stddev=FLAGS.bias_init_stddev)
attn_mlp_fn = partial(make_mlp_model,
                      FLAGS.latent_dim,
                      FLAGS.attn_output_dim,
                      FLAGS.num_layers,
                      activation=tf.nn.leaky_relu,
                      l2_regularizer_weight=0.000001,
                      bias_init_stddev=FLAGS.bias_init_stddev)

singlehead_my_attn = partial(self_attn_gnn,
                             kq_dim=FLAGS.attn_kq_dim,
                             v_dim=FLAGS.attn_v_dim,
                             make_mlp_fn=make_mlp_fn,
                             kq_dim_division=True)
multihead_my_attn = partial(
    multihead_self_attn_gnn,
    kq_dim=FLAGS.attn_kq_dim,
    v_dim=FLAGS.attn_v_dim,
    concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
    make_mlp_fn=make_mlp_fn,
    num_heads=FLAGS.attn_num_heads,
    kq_dim_division=True,
    layer_norm=FLAGS.use_layer_norm)
dm_attn = partial(dm_self_attn_gnn,
                  kq_dim=FLAGS.attn_kq_dim,
                  v_dim=FLAGS.attn_v_dim,
                  make_mlp_fn=make_mlp_fn,
                  num_heads=FLAGS.attn_num_heads,
                  concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
                  kq_dim_division=True,
                  layer_norm=FLAGS.use_layer_norm)
latest_attn = partial(
    latest_self_attention_gnn,
    kq_dim=FLAGS.attn_kq_dim,
    v_dim=FLAGS.attn_v_dim,
    concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
    make_mlp_fn=make_mlp_fn,
    train_batch_size=FLAGS.train_batch_size,
    max_n_node=max(dataset.full_n_nodes()),
    num_heads=FLAGS.attn_num_heads,
    kq_dim_division=True)
additive_attn = partial(padded_additive_self_attn_gnn,
                        v_dim=FLAGS.attn_v_dim,
                        attn_mlp_fn=attn_mlp_fn,
                        attn_output_dim=FLAGS.attn_output_dim,
                        gnn_mlp_fn=make_mlp_fn,
                        max_n_node=max(dataset.full_n_nodes()),
                        train_batch_size=FLAGS.train_batch_size,
                        node_embedding_dim=FLAGS.node_embedding_dim)
ATTN_MAP = {
    'singlehead_my_attn': singlehead_my_attn,
    'multihead_my_attn': multihead_my_attn,
    'dm_attn': dm_attn,
    'latest_attn': latest_attn,
    'additive_attn': additive_attn,
}
grevnet = GRevNet(
    ATTN_MAP[FLAGS.attn_type],
    FLAGS.num_coupling_layers,
    FLAGS.node_embedding_dim,
    use_batch_norm=FLAGS.use_batch_norm,
    weight_sharing=FLAGS.weight_sharing) if not FLAGS.use_gnf else GNFBlock(
        ATTN_MAP[FLAGS.attn_type], FLAGS.num_coupling_layers,
        FLAGS.node_embedding_dim, FLAGS.use_batch_norm, FLAGS.weight_sharing,
        FLAGS.use_efficient_backprop)

grevnet_reverse_output, log_det_jacobian = grevnet(graphs_tuple, inverse=True)
grevnet_output_norm = tf.norm(grevnet_reverse_output.nodes, axis=1)
mvn = tfd.MultivariateNormalDiag(tf.zeros(FLAGS.node_embedding_dim),
                                 tf.ones(FLAGS.node_embedding_dim))

log_prob_zs = tf.reduce_sum(mvn.log_prob(grevnet_reverse_output.nodes))
log_prob_xs = log_prob_zs + log_det_jacobian
total_loss = -1 * log_prob_xs
per_node_loss = total_loss / tf.cast(tf.reduce_sum(graphs_tuple.n_node),
                                     tf.float32)
# Optimizer.
global_step = tf.Variable(0, trainable=False, name='global_step')
lr = None
if FLAGS.lr_type == 'constant':
    lr = FLAGS.lr
elif FLAGS.lr_type == 'fixed_decay':
    lr = tf.train.exponential_decay(learning_rate=FLAGS.lr,
                                    global_step=global_step,
                                    decay_steps=FLAGS.lr_fixed_decay_steps,
                                    decay_rate=FLAGS.lr_fixed_decay_rate,
                                    staircase=FLAGS.lr_fixed_decay_staircase)
elif FLAGS.lr_type == 'polynomial_decay':
    lr = tf.train.polynomial_decay(learning_rate=FLAGS.lr,
                                   global_step=global_step,
                                   decay_steps=FLAGS.num_train_iters,
                                   end_learning_rate=FLAGS.lr / 100,
                                   power=0.5)
elif FLAGS.lr_type == 'schedule':
    lr = tf.placeholder(tf.float32, [], name='lr_schedule_placeholder')
optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                   beta1=FLAGS.adam_beta1,
                                   beta2=FLAGS.adam_beta2,
                                   epsilon=FLAGS.adam_epsilon)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    grads_and_vars = optimizer.compute_gradients(per_node_loss)
    if FLAGS.clip_gradient_by_value:
        grads_and_vars = [
            (tf.clip_by_value(grad, FLAGS.clip_gradient_value_lower,
                              FLAGS.clip_gradient_value_upper), var)
            for grad, var in grads_and_vars
        ]

    if FLAGS.clip_gradient_by_norm:
        grads_and_vars = [(tf.clip_by_norm(grad,
                                           FLAGS.clip_gradient_norm), var)
                          for grad, var in grads_and_vars]

    step_op = optimizer.apply_gradients(grads_and_vars,
                                        global_step=global_step)


# Sample model.
sample_n_node_placeholder = tf.placeholder(tf.int32,
                                           shape=[FLAGS.sample_size],
                                           name="sample_n_node_placeholder")
sample_nodes = mvn.sample(
    sample_shape=(tf.reduce_sum(sample_n_node_placeholder, )))
sample_log_prob = mvn.log_prob(sample_nodes)
sample_edges, sample_globals, sample_receivers, sample_senders, sample_n_edge = transform_example(
    sample_n_node_placeholder)
sample_graphs_tuple = gn.graphs.GraphsTuple(nodes=sample_nodes,
                                            edges=sample_edges,
                                            globals=sample_globals,
                                            receivers=sample_receivers,
                                            senders=sample_senders,
                                            n_node=sample_n_node_placeholder,
                                            n_edge=sample_n_edge)

sample_grevnet_top = grevnet(sample_graphs_tuple, inverse=False)
sample_pred_adj = pred_adj(sample_grevnet_top,
                           distance_fn=scaled_hacky_sigmoid_l2)

tf.summary.scalar('total_loss', total_loss)
tf.summary.scalar('per_node_loss', per_node_loss)
tf.summary.scalar('log_prob_xs', log_prob_xs)
tf.summary.scalar('log_prob_zs', log_prob_zs)
tf.summary.scalar('log_det_jacobian', log_det_jacobian)

merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = reset_sess(config)

#saver = tf.train.Saver()
#saver.restore(sess, "/scratch/gobi2/jyliu/test_runs/test_grevnet_save_model/checkpoints-6831")

#if FLAGS.debug:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'train'), sess.graph)
eval_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'test'), sess.graph)

flags_map = tf.app.flags.FLAGS.flag_values_dict()
with open(os.path.join(LOGDIR, 'desc.txt'), 'w') as f:
    for (k, v) in flags_map.items():
        f.write("{}: {}\n".format(k, str(v)))

saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)
train_values = {}
values_map = {
    "merge": merged,
    "step_op": step_op,
    "total_loss": total_loss,
    "per_node_loss": per_node_loss,
    "log_prob_zs": log_prob_zs,
    "log_prob_xs": log_prob_xs,
    "log_det_jacobian": log_det_jacobian,
    "graphs_tuple": graphs_tuple,
    "batch_n_node": batch_n_node,
}
if FLAGS.debug_grads:
    for g, v in grads_and_vars:
        values_map[v.name] = g

samples_map = {
    "sample_pred_adj": sample_pred_adj,
    "sample_grevnet_top": sample_grevnet_top,
    "sample_log_prob": sample_log_prob,
    "sample_grevnet_top_nodes": sample_grevnet_top.nodes,
    "sample_nodes": sample_nodes,
    "sample_n_node": sample_n_node_placeholder,
}

feed_dict = {}
dataset_generator = None
if FLAGS.variable_dataset:
    dataset_generator = GrevnetDatasetVariable(
        os.path.join(logdir_prefix, FLAGS.train_data_dir), FLAGS.max_nodes)
else:
    dataset_generator = GrevnetDatasetFixed(
        os.path.join(logdir_prefix, FLAGS.train_data_dir),
        FLAGS.train_batch_size)
for iteration in range(0, FLAGS.num_train_iters + 1):
    node_embeddings, n_node = dataset_generator.train_batch()
    feed_dict = {
        node_embeddings_placeholder: node_embeddings,
        n_node_placeholder: n_node
    }
    train_values = sess.run(values_map, feed_dict=feed_dict)
    if train_writer and (iteration % FLAGS.summary_every_n_steps == 0):
        train_writer.add_summary(train_values['merge'], iteration)
    if iteration % FLAGS.log_every_n_steps == 0:
        logger.info("*" * 100)
        logger.info("iteration num: {}".format(iteration))
        logger.info("total loss: {}".format(train_values["total_loss"]))
        logger.info("per node loss: {}".format(train_values["per_node_loss"]))
        logger.info("log prob zs: {}".format(train_values["log_prob_zs"]))
        logger.info("log det jacobian: {}".format(
            train_values["log_det_jacobian"]))
        logger.info("batch {}, tot node {}, size {}".format(
            train_values["graphs_tuple"].n_node, train_values["batch_n_node"],
            len(train_values["graphs_tuple"].n_node),
            len(train_values["graphs_tuple"].n_node)))

    #for g, v in grads_and_vars:
        #logger.info("grad of {}: {}".format(v.name, train_values[v.name]))
    if FLAGS.debug_grads:
        outfile = os.path.join(grads_folder, "iter_{}.p".format(iteration))
        m = {}
        for g, v in grads_and_vars:
            m[v.name] = train_values[v.name]
        with open(outfile, 'wb') as f:
            pickle.dump(m, f)
    # Save model.
    if iteration % FLAGS.save_every_n_steps == 0:
        saver.save(sess,
                   os.path.join(LOGDIR, 'checkpoints'),
                   global_step=global_step)

    # Write out graphs.
    if iteration % FLAGS.write_graphs_every_n_steps == 0 and iteration > FLAGS.write_graphs_min_iter:
        graphs_dir = os.path.join(GRAPHS_LOGDIR, "iter_{}".format(iteration))
        os.makedirs(graphs_dir)
        feed_dict = {
            sample_n_node_placeholder:
            random.sample(FLAGS.sample_size * dataset.test_n_nodes(),
                          FLAGS.sample_size)
        }
        logger.info("*" * 100)
        logger.info("iteration num: {}".format(iteration))
        print("writing graphs...")
        graphs = []
        values = sess.run(samples_map, feed_dict=feed_dict)
        n_node = values["sample_grevnet_top"].n_node
        sample_log_prob = values["sample_log_prob"]
        pred_adj = values["sample_pred_adj"]
        adjacency = np.where(pred_adj > 0.5, np.ones_like(pred_adj),
                             np.zeros_like(pred_adj))
        n_node_cum = np.cumsum(n_node)
        start_ind = 0
        for i in range(FLAGS.sample_size):
            end_ind = n_node_cum[i]
            num_nodes = end_ind - start_ind
            graph = adjacency[start_ind:end_ind, start_ind:end_ind]
            graph = nx.convert_matrix.from_numpy_matrix(graph)
            single_sample_log_prob = np.mean(
                sample_log_prob[start_ind:end_ind])
            visualize_graph(graph,
                            filename=os.path.join(
                                graphs_dir,
                                "graph_{}_prob_{:.2f}_nnode_{}.png".format(
                                    i, single_sample_log_prob, num_nodes)))
            graphs.append(graph)
            start_ind = end_ind
        pickle.dump(
            graphs,
            open(os.path.join(graphs_dir, "pickled.p".format(iteration)),
                 'wb'))
        logger.info("done writing graphs")
