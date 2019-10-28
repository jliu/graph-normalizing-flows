from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from itertools import permutations
import collections
import logging
import math
import os
import pickle
import random
import warnings

from sklearn import datasets
from sklearn.manifold import spectral_embedding
from sonnet.python.modules import base
from absl import flags
import graph_nets as gn
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
#import tfplot
tfd = tfp.distributions
tfb = tfp.bijectors

from gnn import *
from grevnet import *
from grevnet_synthetic_data import *
from utils import *
warnings.filterwarnings("ignore")

# Graph params.
flags.DEFINE_integer('node_embedding_dim', 2,
                     'Number of dimensions in node embeddings.')
flags.DEFINE_integer('min_nodes', 100, 'Min nodes in graph.')
flags.DEFINE_integer('max_nodes', 101, 'Max nodes in graph.')
flags.DEFINE_string('generate_graphs_fn', 'fc', 'Can be fc or isolated.')

# GRevNet params.
flags.DEFINE_bool('use_efficient_backprop', True, '')
flags.DEFINE_bool('use_gnf', True, '')
flags.DEFINE_integer(
    'num_coupling_layers', 12,
    'Number of coupling layers in GRevNet. Each coupling layers '
    'consists of applying F and then G, where F and G are GNNs.')
flags.DEFINE_bool('weight_sharing', False, '')

# GNN params.
flags.DEFINE_string(
    'make_gnn_fn', 'dm_self_attn', 'Function that makes a GNN'
    'of a specific type. Options are make_gru_gnn, '
    'make_independent_gnn, make_avg_then_mlp_gnn')
flags.DEFINE_integer('gnn_num_layers', 5,
                     'Number of layers to use in MLP in GRevNet.')
flags.DEFINE_integer('gnn_latent_dim', 256,
                     'Latent dim for GNN used in GRevNet.')
flags.DEFINE_float('gnn_bias_init_stddev', 0.1,
                   'Used to initialize biases in GRevNet MLPs.')
flags.DEFINE_float(
    'gnn_l2_regularizer_weight', 0.1,
    'How much to weight the L2 regularizer for the GNN MLP weights.')
flags.DEFINE_float(
    'gnn_avg_then_mlp_epsilon', 1.0,
    'How much to weight the current node embeddings compared to the aggregate'
    'of its neighbors.')

# Self-attention params.
flags.DEFINE_integer('attn_kq_dim', 10, '')
flags.DEFINE_integer('attn_v_dim', 10, '')
flags.DEFINE_integer('attn_num_heads', 8, '')
flags.DEFINE_integer('attn_concat_heads_output_dim', 80, '')
flags.DEFINE_bool('attn_concat', True, '')
flags.DEFINE_bool('attn_residual', False, '')
flags.DEFINE_bool('attn_layer_norm', False, '')

# Training params.
flags.DEFINE_bool('use_lr_schedule', False, '')
flags.DEFINE_integer('lr_schedule_ramp_up', 1000, '')
flags.DEFINE_integer('lr_schedule_hold', 2000, '')

flags.DEFINE_bool('smaller_stddev_samples', False, '')
flags.DEFINE_float('smaller_stddev', 0.5, '')

flags.DEFINE_bool('use_batch_norm', True,
                  'Whether to use batch norm during training.')
flags.DEFINE_string('dataset', None, 'Which dataset to use.')
flags.DEFINE_string('logdir', 'test_runs/test_grevnet',
                    'Where to write training files.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size used at training.')
flags.DEFINE_integer('num_train_iters', 15000,
                     'Number of steps to run training.')
flags.DEFINE_integer('save_every_n_steps', 10000, 'How often to save model.')
flags.DEFINE_integer('log_every_n_steps', 50, 'How often to log model stats.')
flags.DEFINE_integer('write_imgs_every_n_steps', 1000,
                     'How often to log model stats.')
flags.DEFINE_integer('write_data_every_n_steps', 10000,
                     'How often to log model stats.')
flags.DEFINE_integer('max_checkpoints_to_keep', 5,
                     'Max model checkpoints to save.')
flags.DEFINE_integer('max_individual_samples', 10,
                     'Max individual samples to display in Tensorboard.')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_bool('include_histograms', False, '')
flags.DEFINE_bool('add_optimizer_summaries', False, '')
flags.DEFINE_bool('add_weight_summaries', False, '')

# Optimizer params.
flags.DEFINE_float('lr', 1e-04, 'Learning rate.')
flags.DEFINE_bool('use_lr_decay', True, 'Whether to decay learning rate.')
flags.DEFINE_integer('lr_decay_steps', 1000,
                     'How often to decay learning rate.')
flags.DEFINE_float('lr_decay_rate', 0.96, 'How much to decay learning rate.')
flags.DEFINE_float('adam_beta1', 0.9, 'Adam optimizer beta1.')
flags.DEFINE_float('adam_beta2', 0.9, 'Adam optimizer beta2.')
flags.DEFINE_float('adam_epsilon', 1e-08, 'Adam optimizer epsilon.')
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
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)
tf.random.set_random_seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})

MAX_SEED = 2**32 - 1
MIN_MAX_NODES = (FLAGS.min_nodes, FLAGS.max_nodes)
logdir_prefix = os.environ.get('MLPATH')
if not logdir_prefix:
    logdir_prefix = '.'
LOGDIR = os.path.join(logdir_prefix, FLAGS.logdir)
imgs_dir = os.path.join(LOGDIR, 'imgs')
pickle_dir = os.path.join(LOGDIR, 'pickle_files')
os.makedirs(imgs_dir)
os.makedirs(pickle_dir)

DATASET = DATASETS_MAP[FLAGS.dataset]


def make_avg_concat_then_mlp_gnn():
    return avg_concat_then_mlp_gnn(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev))


def make_sum_concat_then_mlp_gnn():
    return sum_concat_then_mlp_gnn(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev))


def make_gru_gnn():
    gru_block = GRUBlock(FLAGS.node_embedding_dim / 2)
    return NodeBlockGNN(gru_block)


def make_avg_then_mlp_gnn():
    return avg_then_mlp_gnn(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev), FLAGS.gnn_avg_then_mlp_epsilon)


def make_independent_gnn():
    return gn.modules.GraphIndependent(node_model_fn=partial(
        make_mlp_model, FLAGS.gnn_latent_dim, FLAGS.node_embedding_dim /
        2, FLAGS.gnn_num_layers, tf.nn.leaky_relu,
        FLAGS.gnn_l2_regularizer_weight, FLAGS.gnn_bias_init_stddev))


def make_experimental_gnn():
    return ExperimentalGNN(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev), FLAGS.gnn_avg_then_mlp_epsilon,
        FLAGS.train_batch_size)


def make_dm_self_attn_gnn():
    return dm_self_attn_gnn(
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        make_mlp_fn=partial(make_mlp_model, FLAGS.gnn_latent_dim,
                            FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                            tf.nn.relu, FLAGS.gnn_l2_regularizer_weight,
                            FLAGS.gnn_bias_init_stddev),
        num_heads=FLAGS.attn_num_heads,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        concat=FLAGS.attn_concat,
        residual=FLAGS.attn_residual,
        layer_norm=FLAGS.attn_layer_norm)


def make_self_attn_gnn():
    return self_attn_gnn(kq_dim=FLAGS.attn_kq_dim,
                         v_dim=FLAGS.attn_v_dim,
                         make_mlp_fn=partial(make_mlp_model,
                                             FLAGS.gnn_latent_dim,
                                             FLAGS.node_embedding_dim / 2,
                                             FLAGS.gnn_num_layers, tf.nn.relu,
                                             FLAGS.gnn_l2_regularizer_weight,
                                             FLAGS.gnn_bias_init_stddev),
                         kq_dim_division=True)


def make_multihead_self_attn_gnn():
    return multihead_self_attn_gnn(
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        make_mlp_fn=partial(make_mlp_model, FLAGS.gnn_latent_dim,
                            FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                            tf.nn.relu, FLAGS.gnn_l2_regularizer_weight,
                            FLAGS.gnn_bias_init_stddev),
        num_heads=FLAGS.attn_num_heads,
        kq_dim_division=True)


def make_latest_self_attn_gnn():
    return latest_self_attention_gnn(
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        make_mlp_fn=partial(make_mlp_model, FLAGS.gnn_latent_dim,
                            FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                            tf.nn.relu, FLAGS.gnn_l2_regularizer_weight,
                            FLAGS.gnn_bias_init_stddev),
        train_batch_size=FLAGS.train_batch_size,
        max_n_node=6,
        num_heads=FLAGS.attn_num_heads,
        kq_dim_division=True)


MAKE_GNN_FN_MAP = {
    'gru': make_gru_gnn,
    'avg_then_mlp': make_avg_then_mlp_gnn,
    'avg_concat_then_mlp': make_avg_concat_then_mlp_gnn,
    'sum_concat_then_mlp': make_sum_concat_then_mlp_gnn,
    'independent': make_independent_gnn,
    'experimental': make_experimental_gnn,
    'dm_self_attn': make_dm_self_attn_gnn,
    'self_attn': make_self_attn_gnn,
    'multihead_self_attn': make_multihead_self_attn_gnn,
    'latest_self_attn': make_latest_self_attn_gnn,
}
MAKE_GNN_FN = MAKE_GNN_FN_MAP[FLAGS.make_gnn_fn]

# Data placeholders.
data_dict = DATASET.get_next_batch_data_dicts(FLAGS.train_batch_size)
graph_phs = gn.utils_tf.placeholders_from_data_dicts(
    DATASET.get_next_batch_data_dicts(FLAGS.train_batch_size))
graph_phs.n_node.set_shape([FLAGS.train_batch_size])

single_training_graph = gn.utils_tf.get_graph(
    graph_phs, np.random.randint(FLAGS.train_batch_size))

grevnet = GRevNet(
    MAKE_GNN_FN,
    FLAGS.num_coupling_layers,
    FLAGS.node_embedding_dim,
    use_batch_norm=FLAGS.use_batch_norm,
    weight_sharing=FLAGS.weight_sharing) if not FLAGS.use_gnf else GNFBlock(
        make_gnn_fn=MAKE_GNN_FN,
        num_timesteps=FLAGS.num_coupling_layers,
        node_embedding_dim=FLAGS.node_embedding_dim,
        use_batch_norm=FLAGS.use_batch_norm,
        weight_sharing=FLAGS.weight_sharing,
        use_efficient_backprop=FLAGS.use_efficient_backprop)

# Log prob(z).
grevnet_reverse_output, log_det_jacobian = grevnet(graph_phs, inverse=True)
mvn = tfd.MultivariateNormalDiag(tf.zeros(FLAGS.node_embedding_dim),
                                 tf.ones(FLAGS.node_embedding_dim))
log_prob_zs = tf.reduce_sum(mvn.log_prob(grevnet_reverse_output.nodes))
log_prob_xs = log_prob_zs + log_det_jacobian
total_loss = -1 * log_prob_xs

num_nodes = tf.cast(tf.reduce_sum(graph_phs.n_node), tf.float32)
loss_per_node = total_loss / num_nodes
log_prob_xs_per_node = log_prob_xs / num_nodes
log_prob_zs_per_node = log_prob_zs / num_nodes
log_det_jacobian_per_node = log_det_jacobian / num_nodes

mvn = tfd.MultivariateNormalDiag(tf.zeros(FLAGS.node_embedding_dim),
                                 tf.ones(FLAGS.node_embedding_dim))

sample = mvn.sample(sample_shape=(tf.reduce_sum(graph_phs.n_node, )))
sample_graph_phs = graph_phs.replace(nodes=sample)
sample_log_prob = mvn.log_prob(sample)
grevnet_top = grevnet(sample_graph_phs, inverse=False)
grevnet_top_nodes = grevnet_top.nodes

#if FLAGS.smaller_stddev_samples:
#    smaller_mvn = tfd.MultivariateNormalDiag(
#        tf.zeros(FLAGS.node_embedding_dim),
#        tf.zeros(FLAGS.node_embedding_dim) + FLAGS.smaller_stddev)
#    smaller_sample = smaller_mvn.sample(
#        sample_shape=(tf.reduce_sum(graph_phs.n_node, )))
#    smaller_sample_graph_phs = graph_phs.replace(nodes=smaller_sample)
#    smaller_grevnet_top = grevnet(smaller_sample_graph_phs, inverse=False)
#    smaller_grevnet_top_nodes = smaller_grevnet_top.nodes

# Visualize tensors.
#tfplot.summary.plot("training_data", plot_data, [graph_phs.nodes])
#tfplot.summary.plot("single_training_example", plot_data,
#                    [single_training_graph.nodes])
#tfplot.summary.plot("zs", plot_data, [grevnet_reverse_output.nodes])
#tfplot.summary.plot("generated_sample", plot_data, [grevnet_top_nodes])

#if FLAGS.smaller_stddev_samples:
#    tfplot.summary.plot("smaller_variance_sample", plot_data,
#                        [smaller_grevnet_top_nodes])

#for i in range(min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
#    tfplot.summary.plot("single_generated_sample_{}".format(i), plot_data,
#                        [gn.utils_tf.get_graph(grevnet_top, i).nodes])
#
#if FLAGS.smaller_stddev_samples:
#    for i in range(min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
#        tfplot.summary.plot(
#            "smaller_single_generated_sample_{}".format(i), plot_data,
#            [gn.utils_tf.get_graph(smaller_grevnet_top, i).nodes])

# Optimizer.
global_step = tf.Variable(0, trainable=False, name='global_step')
decaying_learning_rate = tf.train.exponential_decay(
    learning_rate=FLAGS.lr,
    global_step=global_step,
    decay_steps=FLAGS.lr_decay_steps,
    decay_rate=FLAGS.lr_decay_rate,
    staircase=False)
learning_rate = decaying_learning_rate if FLAGS.use_lr_decay else FLAGS.lr

learning_rate_placeholder = tf.placeholder(tf.float32, [],
                                           name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder
                                   if FLAGS.use_lr_schedule else learning_rate,
                                   beta1=FLAGS.adam_beta1,
                                   beta2=FLAGS.adam_beta2,
                                   epsilon=FLAGS.adam_epsilon)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    grads_and_vars = optimizer.compute_gradients(total_loss)
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

saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = reset_sess(config)

# Training loss stats.
tf.summary.scalar('total_loss/loss', total_loss)
tf.summary.scalar('total_loss/log_prob_zs', log_prob_zs)
tf.summary.scalar('total_loss/log_det_jacobian', log_det_jacobian)
tf.summary.scalar('total_loss/log_prob_xs', log_prob_xs)
tf.summary.scalar('per_node_loss/loss', loss_per_node)
tf.summary.scalar('per_node_loss/log_prob_zs', log_prob_zs_per_node)
tf.summary.scalar('per_node_loss/log_det_jacobian', log_det_jacobian_per_node)
tf.summary.scalar('per_node_loss/log_prob_xs', log_prob_xs_per_node)

# Optimization stats.
if FLAGS.add_optimizer_summaries:
    for (g, v) in grads_and_vars:
        if g is not None:
            tf.summary.scalar("grads/{}/norm".format(v.name), tf.norm(g))
            tf.summary.scalar("adam_moment_1/{}/norm".format(v.name),
                              tf.norm(optimizer.get_slot(v, 'm')))
            tf.summary.scalar("adam_moment_2/{}/norm".format(v.name),
                              tf.norm(optimizer.get_slot(v, 'v')))
            if FLAGS.include_histograms:
                tf.summary.histogram("grads/{}".format(v.name), g)
                tf.summary.histogram("adam_moment_1/{}".format(v.name),
                                     optimizer.get_slot(v, 'm'))
                tf.summary.histogram("adam_moment_2/{}".format(v.name),
                                     optimizer.get_slot(v, 'v'))

if FLAGS.add_weight_summaries:
    for v in tf.trainable_variables():
        tf.summary.scalar("weights/{}/norm".format(v.name), tf.norm(v))
        if FLAGS.include_histograms:
            tf.summary.histogram("weights/{}".format(v.name), v)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR, sess.graph)

flags_map = tf.app.flags.FLAGS.flag_values_dict()
with open(os.path.join(LOGDIR, 'desc.txt'), 'w') as f:
    for (k, v) in flags_map.items():
        f.write("{}: {}\n".format(k, str(v)))

values_map = {
    "step_op": step_op,
    "graph_phs": graph_phs,
    "grevnet_bottom": grevnet_reverse_output.nodes,
    "total_loss": total_loss,
    "loss_per_node": loss_per_node,
    "merged": merged,
    "grevnet_top_nodes": grevnet_top_nodes,
    "sample_log_prob": sample_log_prob,
    "log_det_jacobian": log_det_jacobian,
}

for i in range(min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
    values_map["generated_sample_{}".format(i)] = gn.utils_tf.get_graph(
        grevnet_top, i).nodes

for iteration in range(FLAGS.num_train_iters + 1):
    feed_dict = {}
    feed_dict[graph_phs] = DATASET.get_next_batch(FLAGS.train_batch_size)
    if FLAGS.use_lr_schedule:
        feed_dict[learning_rate_placeholder] = get_learning_rate(
            iteration, FLAGS.lr, FLAGS.lr_schedule_ramp_up,
            FLAGS.lr_schedule_hold)
    train_values = sess.run(values_map, feed_dict=feed_dict)

    if iteration % FLAGS.save_every_n_steps == 0:
        saver.save(sess,
                   os.path.join(LOGDIR, 'model'),
                   global_step,
                   write_meta_graph=False)

    if iteration % FLAGS.log_every_n_steps == 0:
        writer.add_summary(train_values["merged"], iteration)
        print("*" * 50)
        print("iteration num: {}".format(iteration))
        print("total_loss: {}".format(train_values["total_loss"]))
        print("loss per node: {}".format(train_values["loss_per_node"]))
        print("log det jacobian: {}".format(train_values["log_det_jacobian"]))
        #print("grevnet bottom: {}".format(train_values["grevnet_bottom"]))
        print("original mean {} std dev {}".format(
            np.mean(train_values["graph_phs"].nodes, 0),
            np.std(train_values["graph_phs"].nodes, 0)))
        print("transformed mean {} std dev {}".format(
            np.mean(train_values["grevnet_bottom"], 0),
            np.std(train_values["grevnet_bottom"], 0)))

    if iteration % FLAGS.write_imgs_every_n_steps == 0:
        plot_data(train_values["grevnet_bottom"],
                  os.path.join(imgs_dir, 'iter_{}_zs.png'.format(iteration)))
        plot_data(
            train_values["grevnet_top_nodes"],
            os.path.join(imgs_dir,
                         'iter_{}_aggregated_x.png'.format(iteration)))
        for i in range(
                min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
            name = "generated_sample_{}".format(i)
            outname = "iter_{}_generated_sample_{}.png".format(iteration, i)
            plot_data(train_values[name], os.path.join(imgs_dir, outname))

    #if iteration % FLAGS.write_data_every_n_steps == 0:
    #    pickle.dump(
    #        train_values["grevnet_top_nodes"],
    #        open(
    #            os.path.join(LOGDIR,
    #                         "pickle_files/samples_{}.dat".format(iteration)),
    #            'wb'))
    #    pickle.dump(
    #        train_values["graph_phs"].nodes,
    #        open(
    #            os.path.join(
    #                LOGDIR,
    #                "pickle_files/training_data_{}.dat".format(iteration)),
    #            'wb'))
    #    pickle.dump(
    #        train_values["grevnet_bottom"],
    #        open(
    #            os.path.join(LOGDIR,
    #                         "pickle_files/zs_{}.dat".format(iteration)),
    #            'wb'))
    #    pickle.dump(
    #        train_values["sample_log_prob"],
    #        open(
    #            os.path.join(
    #                LOGDIR,
    #                "pickle_files/sample_log_prob_{}.dat".format(iteration)),
    #            'wb'))
