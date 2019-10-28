from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import hashlib
import logging
import math
import os
import pickle
import sys
import warnings

from absl import flags
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
#import tfplot

from gnn import *
from graph_data import *
from loss import *
from utils import *

warnings.filterwarnings("ignore")

# Attention params.
flags.DEFINE_string('attn_type', '', '')
flags.DEFINE_integer('attn_kq_dim', 64, '')
flags.DEFINE_integer('attn_v_dim', 64, '')
flags.DEFINE_integer('attn_num_heads', 2, '')
flags.DEFINE_integer('attn_concat_heads_output_dim', 64, '')
flags.DEFINE_bool('attn_concat', True, '')
flags.DEFINE_bool('attn_residual', False, '')
flags.DEFINE_bool('attn_layer_norm', False, '')
flags.DEFINE_bool('attn_kq_dim_division', True, '')

# Dataset params.
flags.DEFINE_bool('denoising', False, '')
flags.DEFINE_float('deletion_prob', 0.3, '')
flags.DEFINE_string('dataset', 'graph_rnn_ego', '')
flags.DEFINE_float('split_train_percent', 0.8, '')

# Eval params.
flags.DEFINE_bool('run_eval', True, '')
flags.DEFINE_integer('eval_every_n_steps', 100, '')

# Training params.
flags.DEFINE_string('restore_from_ckpt', '', '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_integer('tf_random_seed', 601904901297, '')
flags.DEFINE_string('logdir', 'test_runs/test_gnn',
                    'Where to write training files.')
flags.DEFINE_integer('train_batch_size', 32, '')
flags.DEFINE_integer('num_train_iters', 200000, '')
flags.DEFINE_integer('log_every_n_steps', 50, '')
flags.DEFINE_integer('summary_every_n_steps', 25, '')
flags.DEFINE_bool('save_trajectories', False, '')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, '')
flags.DEFINE_integer('save_every_n_iter', 10000, '')
flags.DEFINE_integer('incorrect_every_n_iter', 5000, '')
flags.DEFINE_integer('incorrect_history', 5, '')

# Optimizer params.
flags.DEFINE_string(
    'lr_type', 'polynomial_decay',
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

# Loss + distance function.
flags.DEFINE_string('loss_type', 'binary', 'Can be binary or triplet.')
flags.DEFINE_string('binary_dist_fn', 'scaled_hacky_sigmoid_l2', '')
flags.DEFINE_bool('tune_sigmoid', False, '')
flags.DEFINE_bool('use_soft_labels', False, '')

# Triplet loss.
flags.DEFINE_string('triplet_dist_fn', 'l2', '')
flags.DEFINE_string('triplet_adj_dist_fn', 'hacky_sigmoid_l2', '')
flags.DEFINE_string('triplet_loss_fn', 'margin', 'Can be margin or relative.')
flags.DEFINE_integer('num_sampling_loops', 8, '')

# GNN params.
flags.DEFINE_bool('residual', False, '')
flags.DEFINE_bool('weight_sharing', True, '')
flags.DEFINE_bool('use_batch_norm', True, '')
flags.DEFINE_bool('use_layer_norm', False, '')
flags.DEFINE_bool('use_fc_adj_mat', False, '')
flags.DEFINE_bool('bn_test_local_stats', True, '')
flags.DEFINE_integer('num_layers', 3, 'Num of layers of MLP used in GNN.')
flags.DEFINE_integer('latent_dim', 2048, 'Latent dim of MLP used in GNN.')
flags.DEFINE_integer('num_processing_steps', 10,
                     'Number of steps to take in the GNN.')
flags.DEFINE_float('bias_init_stddev', 0.3,
                   'Used for initializing bias weights in GNN.')
flags.DEFINE_float(
    'node_weighting_epsilon', 2.0,
    'How much to weight current node embedding over its neighbors.')

# Node feature params.
flags.DEFINE_integer('node_embedding_dim', 100, 'Dimension of node embeddings.')
flags.DEFINE_float('gaussian_scale', 1.0,
                   'Scale to use for random Gaussian features.')
flags.DEFINE_integer('laplacian_random_seed', 1234,
                     'Random seed used for Laplacian feature generation.')

# Grid graph params.
flags.DEFINE_string('graph_dim', '10,10', '')

# Barabasi-Albert graph params.
flags.DEFINE_integer('barabasi_n', '20', 'Num of nodes in graph.')
flags.DEFINE_integer(
    'barabasi_m', '4',
    'Num of edges to attach from new node to existing nodes.')

flags.DEFINE_integer('skip_gnn_output_norm', 1, '')
flags.DEFINE_bool('print_adj', False, '')

FLAGS = tf.app.flags.FLAGS
logdir_prefix = os.environ.get('MLPATH')
if not logdir_prefix:
    logdir_prefix = '.'
LOGDIR = os.path.join(logdir_prefix, FLAGS.logdir)
os.makedirs(os.path.join(LOGDIR, 'incorrect_edges_figs'))

# Logging and print options.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})
handlers = [logging.StreamHandler(sys.stdout)]
handlers.append(logging.FileHandler(os.path.join(LOGDIR, 'OUTPUT_LOG')))
logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger("logger")

tf.random.set_random_seed(FLAGS.tf_random_seed)
random.seed(FLAGS.random_seed)

temp_const = tf.constant(10.0)
shift_const = tf.constant(1.0)
temp = tf.get_variable(
    'temp', trainable=True, initializer=temp_const,
    dtype=tf.float32) if FLAGS.tune_sigmoid else temp_const
shift = tf.get_variable(
    'shift', trainable=True, initializer=shift_const,
    dtype=tf.float32) if FLAGS.tune_sigmoid else shift_const

TRIPLET_LOSS_FN_MAP = {'margin': margin_loss, 'relative': relative_loss}
DIST_FN_MAP = {
    'exp_l2': exp_l2,
    'sigmoid_dot': sigmoid_dot,
    'hacky_sigmoid_l2': hacky_sigmoid_l2,
    'scaled_hacky_sigmoid_l2': scaled_hacky_sigmoid_l2,
    'hacky_cauchy_l2': hacky_cauchy_l2,
    'dot': dot,
    'l2': l2,
    'sigmoid_l2': partial(sigmoid_l2, temp=temp, shift=shift),
}

binary_dist_fn = DIST_FN_MAP[FLAGS.binary_dist_fn]
triplet_dist_fn = DIST_FN_MAP[FLAGS.triplet_dist_fn]
triplet_adj_dist_fn = DIST_FN_MAP[FLAGS.triplet_adj_dist_fn]
triplet_loss_fn = TRIPLET_LOSS_FN_MAP[FLAGS.triplet_loss_fn]
if FLAGS.denoising:
    dataset = NoisyGraphDataset(FLAGS.dataset, FLAGS.node_embedding_dim,
                                FLAGS.deletion_prob, FLAGS.gaussian_scale)
else:
    dataset = GraphDataset(FLAGS.dataset, FLAGS.node_embedding_dim,
                           FLAGS.gaussian_scale)

# Define GNN and output.
true_graph_phs = gn.utils_tf.placeholders_from_networkxs(
    dataset.train_graphs, force_dynamic_num_graphs=True, name="true_graph_phs")
true_graph_phs.n_node.set_shape([FLAGS.train_batch_size])

noisy_graph_phs = None
if FLAGS.denoising:
    noisy_graph_phs = gn.utils_tf.placeholders_from_networkxs(
        dataset.train_graphs, name="noisy_graph_phs")
make_mlp_fn = partial(make_mlp_model,
                      FLAGS.latent_dim,
                      FLAGS.node_embedding_dim,
                      FLAGS.num_layers,
                      l2_regularizer_weight=FLAGS.l2_regularizer_weight,
                      bias_init_stddev=FLAGS.bias_init_stddev)

dm_self_attn_gnn = partial(
    dm_self_attn_gnn,
    kq_dim=FLAGS.attn_kq_dim,
    v_dim=FLAGS.attn_v_dim,
    make_mlp_fn=make_mlp_fn,
    num_heads=FLAGS.attn_num_heads,
    concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
    concat=FLAGS.attn_concat,
    residual=FLAGS.attn_residual,
    layer_norm=FLAGS.attn_layer_norm,
    kq_dim_division=FLAGS.attn_kq_dim_division)
multihead_self_attn_gnn = partial(
    multihead_self_attn_gnn,
    kq_dim=FLAGS.attn_kq_dim,
    v_dim=FLAGS.attn_v_dim,
    concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
    make_mlp_fn=make_mlp_fn,
    num_heads=FLAGS.attn_num_heads,
    kq_dim_division=FLAGS.attn_kq_dim_division)
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
ATTN_MAP = {
    'dm_attn': dm_self_attn_gnn,
    'latest_attn': latest_attn,
}


is_training = tf.placeholder(tf.bool, name="is_training")
gnn = TimestepGNN(
    ATTN_MAP[FLAGS.attn_type],
    FLAGS.num_processing_steps,
    weight_sharing=FLAGS.weight_sharing,
    use_batch_norm=FLAGS.use_batch_norm,
    residual=FLAGS.residual,
    test_local_stats=FLAGS.bn_test_local_stats,
    use_layer_norm=FLAGS.use_layer_norm)
gnn_output = gnn(noisy_graph_phs if FLAGS.denoising else true_graph_phs,
                 is_training=is_training)

norm_gnn_output = tf.norm(gnn_output.nodes, axis=1)

# Define loss.
true_adj = None
pred_adj = None
distances = None
unreduced_loss = None

if FLAGS.loss_type == 'triplet':
    true_adj, pred_adj, unreduced_loss, distances = triplet_loss(
        gnn_output, true_graph_phs, FLAGS.train_batch_size, triplet_dist_fn,
        triplet_adj_dist_fn, triplet_loss_fn, FLAGS.num_sampling_loops)

elif FLAGS.loss_type == 'binary':
    true_adj, pred_adj, unreduced_loss, sum_loss, mean_loss = binary_loss(
        gnn_output, true_graph_phs, binary_dist_fn, FLAGS.use_soft_labels)

total_incorrect_edges = total_incorrect_edges(true_adj, pred_adj)
incorrect_edges_per_node = total_incorrect_edges / tf.cast(
    tf.reduce_sum(true_graph_phs.n_node), dtype=tf.float32)
incorrect_edges_per_graph = incorrect_edges_per_graph(true_adj, pred_adj,
                                                      true_graph_phs.n_node)
false_positive_edges = false_positive_edges(true_adj, pred_adj)
false_negative_edges = false_negative_edges(true_adj, pred_adj)
regularizer_loss = 0
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
if len(regularization_losses) > 0:
    regularizer_loss = tf.dtypes.cast(tf.add_n(regularization_losses),
                                      tf.float32)
total_loss = sum_loss + regularizer_loss

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
    step_op = optimizer.minimize(total_loss, global_step=global_step)

sum_loss_summary = tf.summary.scalar('sum_loss', sum_loss)
mean_loss_summary = tf.summary.scalar('mean_loss', mean_loss)
total_incorrect_edges_summary = tf.summary.scalar('total_incorrect_edges',
                                                  total_incorrect_edges)
incorrect_edges_per_node_summary = tf.summary.scalar(
    'incorrect_edges_per_node', incorrect_edges_per_node)
false_positive_edges_summary = tf.summary.scalar('false_positive_edges',
                                                 false_positive_edges)
false_negative_edges_summary = tf.summary.scalar('false_negative_edges',
                                                 false_negative_edges)
lr_summary = tf.summary.scalar('learning_rate', lr)

train_summaries = tf.summary.merge_all()
eval_summaries = tf.summary.merge(
    [sum_loss_summary, mean_loss_summary, total_incorrect_edges_summary])

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = reset_sess(config)
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'train'), sess.graph)
eval_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'test'), sess.graph)

flags_map = tf.app.flags.FLAGS.flag_values_dict()
with open(os.path.join(LOGDIR, 'flagfile.txt'), 'w') as f:
    for (k, v) in flags_map.items():
        f.write("--{}={}\n".format(k, str(v)))

output_nodes = []
train_values = {}
if FLAGS.restore_from_ckpt:
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.restore_from_ckpt)
saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)

values_map = {
    "train_summaries": train_summaries,
    "step_op": step_op,
    "true_graph_phs": true_graph_phs,
    "gnn_output": gnn_output.nodes,
    "mean_loss": mean_loss,
    "sum_loss": sum_loss,
    "regularizer_loss": regularizer_loss,
    "true_adj": true_adj,
    "pred_adj": pred_adj,
    "norm_gnn_output": norm_gnn_output,
    "total_incorrect_edges": total_incorrect_edges,
    "incorrect_edges_per_node": incorrect_edges_per_node,
    "incorrect_edges_per_graph": incorrect_edges_per_graph,
    "false_positive_edges": false_positive_edges,
    "false_negative_edges": false_negative_edges,
    "temp": temp,
    "shift": shift,
}

for k, v in values_map.items():
    if k is not "true_graph_phs":
        tf.add_to_collection(k, v)

eval_values_map = {
    "eval_summaries": eval_summaries,
    "eval_mean_loss": mean_loss,
    "eval_sum_loss": sum_loss,
    "eval_total_incorrect_edges": total_incorrect_edges,
    "eval_incorrect_edges_per_node": incorrect_edges_per_node,
    "eval_incorrect_edges_per_graph": incorrect_edges_per_graph,
    "eval_false_positive_edges": false_positive_edges,
    "eval_false_negative_edges": false_negative_edges,
    "eval_true_graph_phs": true_graph_phs,
}
if distances is not None:
    values_map["distances"] = distances

per_graph_num_nodes = np.empty(shape=(FLAGS.incorrect_history *
                                      FLAGS.train_batch_size, ))
per_graph_num_incorrect = np.empty(shape=(FLAGS.incorrect_history *
                                          FLAGS.train_batch_size, ))

for iteration in range(FLAGS.num_train_iters + 1):
    # Run train step.
    feed_dict = {}
    if FLAGS.denoising:
        true_graphs, noisy_graphs, num_removed = dataset.get_next_train_batch(
        FLAGS.train_batch_size)
    else:
        true_graphs = dataset.get_next_train_batch(FLAGS.train_batch_size)
    feed_dict[true_graph_phs] = true_graphs
    if FLAGS.denoising:
        feed_dict[noisy_graph_phs] = noisy_graphs
    feed_dict[is_training] = True
    if FLAGS.lr_type == 'schedule':
        lr_value = get_learning_rate_updated(iteration, FLAGS.lr,
                                             FLAGS.num_train_iters,
                                             FLAGS.lr_schedule_rampup,
                                             FLAGS.lr_schedule_hold)
        feed_dict[lr] = lr_value

    train_values = sess.run(values_map, feed_dict=feed_dict)
    train_values["norm_gnn_output"].sort()
    if FLAGS.save_trajectories:
        output_nodes.append(train_values["gnn_output"])
    if train_writer and (iteration % FLAGS.summary_every_n_steps == 0):
        train_writer.add_summary(train_values['train_summaries'], iteration)
    if iteration % FLAGS.log_every_n_steps == 0:
        logger.info("*" * 100)
        logger.info("iteration num: {}".format(iteration))
        n_node = train_values["true_graph_phs"].n_node
        n_incorrect_per_graph = train_values["incorrect_edges_per_graph"]
        if FLAGS.denoising:
            stats = zip(n_node, num_removed, n_incorrect_per_graph)
        else:
            stats = zip(n_node, n_incorrect_per_graph)

        stats_str = ", ".join(map(lambda x: ":".join(map(str, x)), stats))
        if FLAGS.denoising:
            logger.info("NUM_NODES:NUM_REMOVED:NUM_INCORRECT")
        else:
            logger.info("NUM_NODES:NUM_INCORRECT")
        logger.info(stats_str)
        logger.info("sum loss: {}".format(train_values["sum_loss"]))
        logger.info("mean loss: {}".format(train_values["mean_loss"]))
        logger.info("regularizer loss: {}".format(
            train_values["regularizer_loss"]))
        logger.info("total incorrect edges: {}".format(
            train_values["total_incorrect_edges"]))
        logger.info("incorrect edges per node: {}".format(
            train_values["incorrect_edges_per_node"]))
        logger.info("false positive edges: {}".format(
            train_values["false_positive_edges"]))
        logger.info("false negative edges: {}".format(
            train_values["false_negative_edges"]))
        logger.info("gnn output norm:{}".format(
            np.mean(train_values["norm_gnn_output"])))
        logger.info("temp: {}".format(train_values["temp"]))
        logger.info("shift: {}".format(train_values["shift"]))
        if FLAGS.print_adj:
            logger.info("gnn output:\n{}".format(train_values["gnn_output"]))
            logger.info("true_adj:\n{}".format(train_values["true_adj"]))
            logger.info("pred_adj:\n{}".format(train_values["pred_adj"]))
            if distances is not None:
                logger.info("distances")
                logger.info(train_values["distances"])

    # If needed run eval step.
    if FLAGS.run_eval and iteration % FLAGS.eval_every_n_steps == 0:
        feed_dict = {}
        test_batch = dataset.get_random_test_batch(FLAGS.train_batch_size)
        feed_dict[true_graph_phs] = test_batch
        if FLAGS.denoising:
            feed_dict[noisy_graph_phs] = test_batch
        feed_dict[is_training] = False
        values = sess.run(eval_values_map, feed_dict=feed_dict)
        eval_writer.add_summary(values["eval_summaries"], iteration)
        stats = zip(values["eval_true_graph_phs"].n_node, values["eval_incorrect_edges_per_graph"])
        stats_str = ", ".join(map(lambda x: ":".join(map(str, x)), stats))
        logger.info("*" * 100)
        logger.info("iteration num: {}".format(iteration))
        logger.info("NUM_NODES:NUM_INCORRECT")
        logger.info(stats_str)
        logger.info("eval sum loss: {}".format(values["eval_sum_loss"]))
        logger.info("eval mean loss: {}".format(values["eval_mean_loss"]))
        logger.info("eval total incorrect edges: {}".format(
            values["eval_total_incorrect_edges"]))
        logger.info("eval incorrect edges per node: {}".format(
            values["eval_incorrect_edges_per_node"]))
        logger.info("eval false positive edges: {}".format(
            values["eval_false_positive_edges"]))
        logger.info("eval false negative edges: {}".format(
            values["eval_false_negative_edges"]))

    remainder = iteration % FLAGS.incorrect_every_n_iter
    if remainder < FLAGS.incorrect_history:
        per_graph_num_nodes[
            remainder * FLAGS.train_batch_size:(remainder + 1) *
            FLAGS.train_batch_size] = train_values["true_graph_phs"].n_node
        per_graph_num_incorrect[
            remainder * FLAGS.train_batch_size:(remainder + 1) *
            FLAGS.train_batch_size] = train_values["incorrect_edges_per_graph"]
    elif remainder == FLAGS.incorrect_history:
        plt.scatter(per_graph_num_nodes, per_graph_num_incorrect, c='blue')
        plt.savefig(
            os.path.join(LOGDIR, 'incorrect_edges_figs',
                         '{}.png'.format(iteration)))
        plt.close()

    # Save model.
    if iteration % FLAGS.save_every_n_iter == 0:
        saver.save(sess,
                   os.path.join(LOGDIR, 'checkpoints'),
                   global_step=global_step)

print_adjacency_summary(logger, train_values)
if FLAGS.save_trajectories:
    save_trajectories(output_nodes, LOGDIR)
