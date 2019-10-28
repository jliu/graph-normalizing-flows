from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import graph_nets as gn
import numpy as np
import tensorflow as tf


# Distance functions.
def exp_l2(nodes):
    r = tf.reduce_sum(nodes * nodes, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, tf.transpose(nodes)) + tf.transpose(r)
    s = tf.math.exp(-1 * D)
    return s


def sigmoid_dot(nodes):
    dot = tf.tensordot(nodes, tf.transpose(nodes), axes=1)
    dot * (1 - tf.eye(tf.shape(nodes)[0]))
    return tf.math.sigmoid(dot - 0.5)


def hacky_cauchy_l2(nodes):
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    #TODO: make tunable param
    dist = (1 / math.pi) * tf.atan(-1 * (D - 4)) + 0.5
    return dist


def hacky_sigmoid_l2(nodes):
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    #TODO: make tunable param
    r = 10
    return tf.math.sigmoid(r * (1 - D))


def scaled_hacky_sigmoid_l2(nodes):
    dim = tf.shape(nodes)[1]
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    #TODO: make tunable param
    D /= tf.sqrt(tf.dtypes.cast(dim, tf.float32))
    r = 10
    return tf.math.sigmoid(r * (1 - D))


def sigmoid_l2(nodes, temp, shift):
    dim = tf.shape(nodes)[1]
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    D /= tf.sqrt(tf.dtypes.cast(dim, tf.float32))
    return tf.math.sigmoid(temp * (shift - D))


# Diagonal is all 0's.
def adjacency_matrix(senders, receivers, dim):
    one_hot_senders = tf.one_hot(senders, dim)
    one_hot_receivers = tf.one_hot(receivers, dim)
    adj_mat = tf.einsum('ki,kj->ij', one_hot_senders, one_hot_receivers)
    return adj_mat


def soft_label_adjacency_matrix(senders, receivers, dim, epsilon=0.1):
    one_hot_senders = tf.one_hot(senders, dim)
    one_hot_receivers = tf.one_hot(receivers, dim)
    adj_mat = tf.einsum('ki,kj->ij', one_hot_senders, one_hot_receivers)
    adj_mat = tf.where(tf.greater(adj_mat, 0.5),
                       x=tf.ones_like(adj_mat) - epsilon,
                       y=tf.zeros_like(adj_mat) + epsilon)
    return adj_mat


def remove_diag(square_m):
    dim = tf.shape(square_m)[0]
    return square_m * (1 - tf.eye(dim))


def incorrect_edges_per_graph(true_adj, pred_adj, n_node, abs_tol=0.5):
    diff = remove_diag(tf.math.abs(true_adj - pred_adj))
    num_incorrect = tf.where(tf.greater(diff, abs_tol), tf.ones_like(diff),
                             tf.zeros_like(diff))
    num_incorrect = tf.reduce_sum(num_incorrect, axis=0)
    indices = repeat_1d(tf.range(tf.shape(n_node)[0]), n_node)
    num_incorrect = tf.segment_sum(num_incorrect, indices)
    return tf.cast(num_incorrect / 2, dtype=tf.int32)


def num_incorrect(diff, abs_tol=0.5):
    num_incorrect = tf.where(tf.greater(diff, abs_tol), tf.ones_like(diff),
                             tf.zeros_like(diff))
    return tf.reduce_sum(num_incorrect) / 2


def false_positive_edges(true_adj, pred_adj, abs_tol=0.5):
    diff = remove_diag(pred_adj - true_adj)
    return num_incorrect(diff, abs_tol)


def false_negative_edges(true_adj, pred_adj, abs_tol=0.5):
    diff = remove_diag(true_adj - pred_adj)
    return num_incorrect(diff, abs_tol)


def total_incorrect_edges(true_adj, pred_adj, abs_tol=0.5):
    diff = remove_diag(tf.math.abs(true_adj - pred_adj))
    return num_incorrect(diff, abs_tol)


def repeat_1d(values, repeats):
    idx = tf.concat(
        [tf.constant([0], dtype=tf.int32),
         tf.cumsum(repeats[:-1])], axis=0)
    y = tf.sparse_to_dense(
        sparse_indices=idx,
        output_shape=(tf.reduce_sum(repeats), ),
        sparse_values=values -
        tf.concat([tf.constant([0], dtype=tf.int32), values[:-1]], axis=0))
    return tf.cumsum(y)


def loss_mask(graph):
    def body(i, paddings_left, paddings_right, sizes, output):
        padding = [[0, 0], [paddings_left[i], paddings_right[i]]]
        output = output.write(i, tf.pad(tf.ones(sizes[i]), padding))
        return (i + 1, paddings_left, paddings_right, sizes, output)

    num_graphs = gn.utils_tf.get_num_graphs(graph)
    paddings_left = tf.cumsum(graph.n_node, exclusive=True)
    paddings_right = tf.cumsum(graph.n_node, reverse=True, exclusive=True)
    sizes = tf.stack([graph.n_node, graph.n_node], axis=1)

    loop_condition = lambda i, *_: tf.less(i, num_graphs)
    initial_loop_vars = [
        0, paddings_left, paddings_right, sizes,
        tf.TensorArray(dtype=tf.float32, size=num_graphs, infer_shape=False)
    ]
    _, _, _, _, output = tf.while_loop(loop_condition,
                                       body,
                                       initial_loop_vars,
                                       back_prop=False)
    return output.concat()


def pred_adj(gnn_output, distance_fn):
    lm = loss_mask(gnn_output)
    pred_adj = distance_fn(gnn_output.nodes)
    pred_adj *= lm
    pred_adj = remove_diag(pred_adj)
    return pred_adj


def binary_loss(gnn_output,
                graph_phs,
                distance_fn,
                use_soft_labels=False,
                epsilon=0.1):
    num_nodes = tf.reduce_sum(graph_phs.n_node)
    lm = loss_mask(graph_phs)

    # Compute true and predicted adjacency matrices.
    true_adj = adjacency_matrix(graph_phs.senders, graph_phs.receivers,
                                tf.reduce_sum(graph_phs.n_node))
    if use_soft_labels:
        soft_true_adj = tf.where(tf.greater(true_adj, 0.5),
                                 x=tf.ones_like(true_adj) - epsilon,
                                 y=tf.zeros_like(true_adj) + epsilon)
    pred_adj = distance_fn(gnn_output.nodes)
    pred_adj *= lm

    # Compute element-wise cross entropy.
    ce_loss = tf.keras.backend.binary_crossentropy(
        true_adj if not use_soft_labels else soft_true_adj, pred_adj)
    masked_ce_loss = remove_diag(lm) * ce_loss

    sum_loss = tf.reduce_sum(masked_ce_loss)
    mean_loss = tf.reduce_sum(masked_ce_loss) / tf.dtypes.cast(
        num_nodes**2 - num_nodes, tf.float32)
    return true_adj, pred_adj, ce_loss, sum_loss, mean_loss


def reconstruction_prob(gnn_output, graph_phs, distance_fn):
    num_nodes = tf.reduce_sum(graph_phs.n_node)
    lm = loss_mask(graph_phs)

    # Compute true and predicted adjacency matrices.
    true_adj = soft_label_adjacency_matrix(graph_phs.senders,
                                           graph_phs.receivers,
                                           tf.reduce_sum(graph_phs.n_node))

    pred_adj = distance_fn(gnn_output.nodes)
    pred_adj *= lm
    pred_adj = remove_diag(pred_adj)

    return -1 * tf.losses.log_loss(
        true_adj, pred_adj,
        reduction=tf.losses.Reduction.NONE), pred_adj, true_adj


# Mostly unused distance functions.
def l2(nodes):
    r = tf.reduce_sum(nodes * nodes, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, tf.transpose(nodes)) + tf.transpose(r)
    return D


def dot(nodes):
    return tf.tensordot(nodes, tf.transpose(nodes), axes=1)


def margin_loss(positive_scores, negative_scores, margin=0.65):
    return tf.maximum(margin - positive_scores + negative_scores, 0)


def relative_loss(positive_scores, negative_scores):
    return positive_scores / (positive_scores + negative_scores)


def normalize_probs(m):
    return tf.math.divide(m, tf.reshape(tf.reduce_sum(m, axis=1), [-1, 1]))


def sample_examples(true_adj, pred_adj, n_nodes):
    normalized_probs = normalize_probs(true_adj)
    distro = tf.distributions.Categorical(probs=normalized_probs)
    node_range = tf.range(n_nodes)
    sample = distro.sample()
    indices = tf.stack([node_range, sample], axis=1)
    return tf.gather_nd(pred_adj, indices)


def triplet_loss(gnn_output,
                 graph_phs,
                 batch_size,
                 distance_fn,
                 adjacency_distance_fn,
                 triplet_loss_fn,
                 num_sampling_loops=8):
    n_nodes = tf.reduce_sum(gnn_output.n_node)

    true_adj = adjacency_matrix(graph_phs.senders, graph_phs.receivers,
                                n_nodes)  # diagonal is 0's

    lm = loss_mask(graph_phs)
    inv_true_adj = 1 - true_adj
    inv_true_adj = (1 - tf.eye(n_nodes)) * inv_true_adj  # zero out diagonal
    inv_true_adj = lm * inv_true_adj  # remove entries from other graphs in batch

    distances = distance_fn(gnn_output.nodes)  # diagonal is 1's
    distances = tf.dtypes.cast(distances, tf.float32)
    distances = (1 - tf.eye(n_nodes)) * distances

    total_loss = 0
    for i in range(num_sampling_loops):
        positive_scores = sample_examples(true_adj, distances, n_nodes)
        negative_scores = sample_examples(inv_true_adj, distances, n_nodes)
        total_loss += triplet_loss_fn(positive_scores, negative_scores)
    pred_adj = adjacency_distance_fn(gnn_output.nodes)

    return true_adj, pred_adj, total_loss, distances


def approximate_triplet_loss(gnn_output, graph_phs, batch_size, distance_fn,
                             triplet_loss_fn):
    n_nodes = tf.reduce_sum(gnn_output.n_node)

    true_adj = adjacency_matrix(graph_phs.senders, graph_phs.receivers,
                                n_nodes)  # diagonal is 0's

    lm = loss_mask(graph_phs)
    inv_true_adj = 1 - true_adj
    inv_true_adj = (1 - tf.eye(n_nodes)) * inv_true_adj  # zero out diagonal
    inv_true_adj = lm * inv_true_adj  # remove entries from other graphs in batch

    pred_adj = distance_fn(gnn_output.nodes)  # diagonal is 1's
    pred_adj = tf.dtypes.cast(pred_adj, tf.float32)
    pred_adj = (1 - tf.eye(n_nodes)) * pred_adj

    # TODO: Write the actual loss func.
    return true_adj, pred_adj, total_loss


# Unused functions.
def log_prob_pred_adj(gnn_output, batch_size, distance_fn):
    # Compute true and predicted adjacency matrices.
    true_adj = adjacency_matrix(gnn_output.senders, gnn_output.receivers,
                                tf.reduce_sum(gnn_output.n_node))
    pred_adj = tf.dtypes.cast(distance_fn(gnn_output.nodes), tf.float32)
    pred_adj = tf.log(remove_diag(pred_adj, tf.reduce_sum(gnn_output.n_node)))
    mask = loss_mask(gnn_output)
    masked_pred_adj = tf.math.multiply(mask, pred_adj)

    probs = true_adj * masked_pred_adj + (1 - true_adj) * (1 - masked_pred_adj)
    log_probs = tf.log(probs)
    return tf.reduce_sum(log_probs)
