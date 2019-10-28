import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import tfplot


def visualize_graph(G, filename=None):
    pos = nx.spring_layout(G, k=0.5, iterations=200, scale=100)
    plt.figure(figsize=(30, 10))
    nx.draw(G,
            pos,
            node_color='#e4dce8',
            node_size=50,
            with_labels=False,
            width=2.0)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


# Returns a new TF session.
def reset_sess(config=None):
    try:
        sess.close()
    except NameError:
        pass
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def plot_data(data, outfile):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c='blue')
    plt.savefig(outfile)
    plt.close()


# Plot data as a scatterplot in Tensorboard.
#def plot_data(data):
#    x = data[:, 0]
#    y = data[:, 1]
#    fig, ax = tfplot.subplots(figsize=(4, 3))
#    img = ax.scatter(x, y, c='blue')
#    return img.figure


def plot_num_incorrect_edges_per_graph(n_nodes, n_incorrect):
    fig, ax = tfplot.subplots(figsize=(12, 9))
    img = ax.scatter(n_nodes, n_incorrect, c='blue')
    return img.figure


# Save graph node trajectories.
def save_trajectories(output_nodes, run_name):
    base_path = os.path.join(run_name, 'trajectory_{}')
    trajectories = list(zip(*output_nodes))
    for ind in range(len(trajectories)):
        pickle.dump(trajectories[ind], open(base_path.format(ind + 1), 'wb'))


def print_adjacency_summary(logger, train_values, abs_tol=0.05):
    logger.info(
        "printing true and predicted adj that were not within {}".format(
            abs_tol))
    true_adj = train_values["true_adj"].flatten()
    pred_adj = train_values["pred_adj"].flatten()
    true_to_print = []
    pred_to_print = []
    for true, pred in zip(true_adj, pred_adj):
        if math.isclose(true, pred, abs_tol=0.05):
            continue
        true_to_print.append(true)
        pred_to_print.append(pred)
        if len(true_to_print) == 10:
            logger.info(["{0:0.2f}".format(i) for i in true_to_print])
            logger.info(["{0:0.2f}".format(i) for i in pred_to_print])
            logger.info("*" * 100)
            true_to_print.clear()
            pred_to_print.clear()
    logger.info(["{0:0.2f}".format(i) for i in true_to_print])
    logger.info(["{0:0.2f}".format(i) for i in pred_to_print])


# Learning rate schedule. Linear ramp-up to max_lr, holds for hold_steady
# steps, then sqrt decay from there.
def get_learning_rate(timestep,
                      max_lr,
                      ramp_up=1000,
                      hold_steady=2000,
                      const_multiple=3):
    if timestep < ramp_up:
        return timestep * max_lr / ramp_up
    elif timestep <= hold_steady:
        return max_lr
    else:
        sqrt_diff = math.sqrt(timestep - hold_steady)
        multiplier = min(1 / sqrt_diff, const_multiple / sqrt_diff)
        return multiplier * max_lr


# Learning rate schedule. Linear ramp-up to max_lr, holds for hold_steady
# steps, then sqrt decay from there.
def get_learning_rate_updated(timestep,
                              max_lr,
                              num_train_iters,
                              ramp_up=1000,
                              hold_steady=2000,
                              power=0.5):
    if timestep < ramp_up:
        return timestep * max_lr / ramp_up
    elif timestep <= hold_steady:
        return max_lr
    else:
        target_lr = max_lr / 100
        decay_steps = num_train_iters = ramp_up - hold_steady
        return (max_lr -
                target_lr) * (1 - timestep / decay_steps)**power + target_lr


def get_learning_rate_updated_again(timestep, node_embedding_dim, rampup=1000):
    lr = node_embedding_dim**-0.5 * min(timestep**-0.5,
                                        timestep * rampup**-1.5)
    return lr


def cartesian_graph(a):
    """
    Given at least 2 elements in a, generates the Cartesian product of all
    elements in the list.
    """
    tile_a = tf.expand_dims(
        tf.tile(tf.expand_dims(a[0], 1), [1, tf.shape(a[1])[0]]), 2)
    tile_b = tf.expand_dims(
        tf.tile(tf.expand_dims(a[1], 0), [tf.shape(a[0])[0], 1]), 2)
    cart = tf.concat([tile_a, tile_b], axis=2)
    cart = tf.reshape(cart, [-1, 2])
    for c in a[2:]:
        tile_c = tf.tile(tf.expand_dims(c, 1), [1, tf.shape(cart)[0]])
        tile_c = tf.expand_dims(tile_c, 2)
        tile_c = tf.reshape(tile_c, [-1, 1])
        cart = tf.tile(cart, [tf.shape(c)[0], 1])
        cart = tf.concat([tile_c, cart], axis=1)
    return cart


def permutations(a, times=2):
    """
    Shortcut for generating the Cartesian product of self, using indices so
    that we can work with a small number of elements initially.
    """
    options = tf.range(tf.shape(a)[0])
    indices = cartesian_graph([options for _ in range(times)])
    gathered = tf.gather(a, indices)
    return gathered


def senders_receivers(n_node):
    def body(i, n_node_lower, n_node_cum, output):
        n_node_upper = n_node_cum[i]
        output = output.write(
            i, permutations(tf.range(n_node_lower, n_node_upper)))
        return (i + 1, n_node_cum[i], n_node_cum, output)

    num_graphs = tf.shape(n_node)[0]
    loop_condition = lambda i, *_: tf.less(i, num_graphs)
    initial_loop_vars = [
        0, 0,
        tf.cumsum(n_node),
        tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
    ]
    _, _, _, output = tf.while_loop(loop_condition,
                                    body,
                                    initial_loop_vars,
                                    back_prop=False)
    output = output.concat()
    return output[..., 0], output[..., 1]
