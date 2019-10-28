from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import sqrt

from sonnet.python.modules import base
import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from loss import *

fc = tf.contrib.layers.fully_connected
tfb = tfp.bijectors


# Blocks to update a node's embedding based on its neighbors' embeddings.
class GRUBlock(snt.AbstractModule):
    def __init__(self,
                 output_dim,
                 num_layers=2,
                 latent_dim=256,
                 bias_init_stddev=0.01,
                 agg_fn=tf.unsorted_segment_mean,
                 name="GRUBlock"):
        super(GRUBlock, self).__init__(name=name)
        self.received_edges_aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
            agg_fn)
        self.output_dim = int(output_dim)
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.bias_init_stddev = bias_init_stddev

    def _build(self, graph):
        nodes = graph.nodes
        aggn = self.received_edges_aggregator(graph)
        biases_initializer = tf.initializers.truncated_normal(
            self.bias_init_stddev)

        r_t = tf.nn.relu(
            fc(aggn,
               self.latent_dim,
               activation_fn=None,
               biases_initializer=None) +
            fc(nodes,
               self.latent_dim,
               activation_fn=None,
               biases_initializer=biases_initializer))
        for _ in range(self.num_layers):
            r_t = fc(r_t, self.latent_dim, activation_fn=tf.nn.relu)
        r_t = fc(r_t,
                 self.output_dim,
                 activation_fn=tf.sigmoid,
                 biases_initializer=biases_initializer)

        z_t = tf.nn.relu(
            fc(aggn,
               self.latent_dim,
               activation_fn=None,
               biases_initializer=None) +
            fc(nodes,
               self.latent_dim,
               activation_fn=None,
               biases_initializer=biases_initializer))
        for _ in range(self.num_layers):
            z_t = fc(z_t,
                     self.latent_dim,
                     activation_fn=tf.nn.relu,
                     biases_initializer=biases_initializer)
        z_t = fc(z_t,
                 self.output_dim,
                 activation_fn=tf.sigmoid,
                 biases_initializer=biases_initializer)

        g_t = tf.nn.relu(
            fc(aggn,
               self.output_dim,
               activation_fn=None,
               biases_initializer=None) +
            r_t * fc(nodes,
                     self.output_dim,
                     activation_fn=None,
                     biases_initializer=biases_initializer))
        for _ in range(self.num_layers):
            g_t = fc(g_t,
                     self.latent_dim,
                     activation_fn=tf.nn.relu,
                     biases_initializer=biases_initializer)
        g_t = fc(g_t,
                 self.output_dim,
                 activation_fn=tf.tanh,
                 biases_initializer=biases_initializer)

        new_nodes = (1 - z_t) * g_t + z_t * nodes
        return graph.replace(nodes=new_nodes)


class ConcatThenMLPBlock(snt.AbstractModule):
    def __init__(self, aggn_fn, make_mlp_fn, name="AggThenMLPBlock"):
        super(ConcatThenMLPBlock, self).__init__(name=name)
        self._received_edges_aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
            aggn_fn)
        self._mlp = make_mlp_fn()

    def _build(self, graph):
        nodes = tf.concat(
            [graph.nodes, self._received_edges_aggregator(graph)], axis=1)
        nodes = self._mlp(nodes)
        return graph.replace(nodes=nodes)


class AggThenMLPBlock(snt.AbstractModule):
    def __init__(self, aggn_fn, make_mlp_fn, epsilon, name="AggThenMLPBlock"):
        super(AggThenMLPBlock, self).__init__(name=name)
        self._received_edges_aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
            aggn_fn)
        self._mlp = make_mlp_fn()
        self.epsilon = epsilon

    def _build(self, graph):
        nodes = self.epsilon * graph.nodes + self._received_edges_aggregator(
            graph)
        nodes = self._mlp(nodes)
        return graph.replace(nodes=nodes)


# GNN that updates node embeddings based on the neighbor node embeddings.
class IdentityModule(base.AbstractModule):
    def _build(self, inputs):
        return tf.identity(inputs)


EDGE_BLOCK_OPT = {
    "use_edges": False,
    "use_receiver_nodes": False,
    "use_sender_nodes": True,
    "use_globals": False,
}


class NodeBlockGNN(snt.AbstractModule):
    def __init__(self,
                 node_block,
                 edge_block_opt=EDGE_BLOCK_OPT,
                 name="NodeBlockGNN"):
        super(NodeBlockGNN, self).__init__(name=name)

        with self._enter_variable_scope():
            self._edge_block = gn.blocks.EdgeBlock(
                edge_model_fn=IdentityModule, **EDGE_BLOCK_OPT)
            self._node_block = node_block

    def _build(self, graph):
        return self._node_block(self._edge_block(graph))


def make_mlp_model(latent_dim,
                   output_dim,
                   num_layers,
                   activation=tf.nn.relu,
                   l2_regularizer_weight=0.01,
                   bias_init_stddev=0.1):
    layers = [latent_dim] * (num_layers - 1)
    layers.append(output_dim)
    return snt.Sequential([
        snt.nets.MLP(
            layers,
            activation=activation,
            initializers={
                'w': tf.initializers.glorot_normal(),
                'b': tf.initializers.truncated_normal(stddev=bias_init_stddev),
            },
            #regularizers={
                #'w': tf.contrib.layers.l2_regularizer(l2_regularizer_weight),
                #'b': tf.contrib.layers.l2_regularizer(l2_regularizer_weight)
            #},
            activate_final=False),
    ])


class TimestepGNN(snt.AbstractModule):
    """Runs the input GNN for num_processing_steps # of timesteps.
    """

    def __init__(self,
                 make_gnn_fn,
                 num_timesteps,
                 weight_sharing=False,
                 use_batch_norm=False,
                 residual=True,
                 test_local_stats=False,
                 use_layer_norm=False,
                 name="TimestepGNN"):
        super(TimestepGNN, self).__init__(name=name)
        self._weight_sharing = weight_sharing
        self._num_timesteps = num_timesteps
        self._use_batch_norm = use_batch_norm
        self._residual = residual
        self._bns = []
        self._lns = []
        self._test_local_stats = test_local_stats
        self._use_layer_norm = use_layer_norm
        with self._enter_variable_scope():
            if not weight_sharing:
                self._gnn = [make_gnn_fn() for _ in range(num_timesteps)]
            else:
                self._gnn = make_gnn_fn()
            if use_batch_norm:
                self._bns = [
                    snt.BatchNorm(scale=True) for _ in range(num_timesteps)
                ]
            if use_layer_norm:
                self._lns = [snt.LayerNorm() for _ in range(num_timesteps)]

    def _build(self, input_op, is_training):
        output = input_op
        for i in range(self._num_timesteps):
            if self._use_batch_norm:
                norm_nodes = self._bns[i](
                    output.nodes,
                    is_training=is_training,
                    test_local_stats=self._test_local_stats)
                output = output.replace(nodes=norm_nodes)
            if self._use_layer_norm:
                norm_nodes = self._lns[i](output.nodes)
                output = output.replace(nodes=norm_nodes)
            if not self._weight_sharing:
                output = self._gnn[i](output)
            else:
                output = self._gnn(output)
        if self._residual:
            output = output.replace(nodes=output.nodes + input_op.nodes)
        return output


def avg_then_mlp_gnn(make_mlp_fn, epsilon):
    avg_then_mlp_block = AggThenMLPBlock(tf.unsorted_segment_mean, make_mlp_fn,
                                         epsilon)
    return NodeBlockGNN(avg_then_mlp_block)


def sum_then_mlp_gnn(make_mlp_fn, epsilon):
    sum_then_mlp_block = AggThenMLPBlock(tf.unsorted_segment_sum, make_mlp_fn,
                                         epsilon)
    return NodeBlockGNN(sum_then_mlp_block)


def sum_concat_then_mlp_gnn(make_mlp_fn):
    node_block = ConcatThenMLPBlock(tf.unsorted_segment_sum, make_mlp_fn)
    return NodeBlockGNN(node_block)


def avg_concat_then_mlp_gnn(make_mlp_fn):
    node_block = ConcatThenMLPBlock(tf.unsorted_segment_mean, make_mlp_fn)
    return NodeBlockGNN(node_block)


def make_batch_norm():
    bn = tf.layers.BatchNormalization(
        axis=-1, gamma_constraint=lambda x: tf.nn.relu(x) + 1e-6)
    return tfb.BatchNormalization(batchnorm_layer=bn, training=True)


def get_gnns(num_timesteps, make_gnn_fn):
    return [make_gnn_fn() for _ in range(num_timesteps)]


def print_variable(v, v_name):
    return tf.Print(v, [v], "{} is: ".format(v_name), summarize=1000, first_n=1)

class GRevNet(snt.AbstractModule):
    def __init__(self,
                 make_gnn_fn,
                 num_timesteps,
                 node_embedding_dim,
                 use_batch_norm=False,
                 weight_sharing=False,
                 name="GRevNet"):
        super(GRevNet, self).__init__(name=name)
        self.num_timesteps = num_timesteps
        self.weight_sharing = weight_sharing
        if weight_sharing:
            self.s = [make_gnn_fn(), make_gnn_fn()]
            self.t = [make_gnn_fn(), make_gnn_fn()]
            #self.s = [
            #    get_gnns(num_timesteps, make_gnn_fn),
            #    get_gnns(num_timesteps, make_gnn_fn)
            #]
        else:
            self.s = [
                get_gnns(num_timesteps, make_gnn_fn),
                get_gnns(num_timesteps, make_gnn_fn)
            ]
            self.t = [
                get_gnns(num_timesteps, make_gnn_fn),
                get_gnns(num_timesteps, make_gnn_fn)
            ]
        self.use_batch_norm = use_batch_norm
        self.bns = [[make_batch_norm() for _ in range(num_timesteps)],
                    [make_batch_norm() for _ in range(num_timesteps)]]

    def f(self, x):
        log_det_jacobian = 0
        x0, x1 = tf.split(x.nodes, num_or_size_splits=2, axis=1)
        x0 = x.replace(nodes=x0)
        x1 = x.replace(nodes=x1)
        for i in range(self.num_timesteps):
            if self.use_batch_norm:
                bn = self.bns[0][i]
                log_det_jacobian += bn.inverse_log_det_jacobian(x0.nodes, 2)
                x0 = x0.replace(nodes=bn.inverse(x0.nodes))
            if self.weight_sharing:
                #s = self.s[0][i](x0).nodes
                #t = self.s[0][i](x0).nodes
                s = self.s[0](x0).nodes
                t = self.t[0](x0).nodes
            else:
                s = self.s[0][i](x0).nodes
                t = self.t[0][i](x0).nodes
            log_det_jacobian += tf.reduce_sum(s)
            x1 = x1.replace(nodes=x1.nodes * tf.exp(s) + t)

            if self.use_batch_norm:
                bn = self.bns[1][i]
                log_det_jacobian += bn.inverse_log_det_jacobian(x1.nodes, 2)
                x1 = x1.replace(nodes=bn.inverse(x1.nodes))
            if self.weight_sharing:
                #s = self.s[1][i](x1).nodes
                #t = self.s[1][i](x1).nodes
                s = self.s[1](x1).nodes
                t = self.t[1](x1).nodes
            else:
                s = self.s[1][i](x1).nodes
                t = self.t[1][i](x1).nodes
            log_det_jacobian += tf.reduce_sum(s)
            x0 = x0.replace(nodes=x0.nodes * tf.exp(s) + t)

        x = x.replace(nodes=tf.concat([x0.nodes, x1.nodes], axis=1))
        return x, log_det_jacobian

    def g(self, z):
        z0, z1 = tf.split(z.nodes, num_or_size_splits=2, axis=1)
        z0 = z.replace(nodes=z0)
        z1 = z.replace(nodes=z1)
        for i in reversed(range(self.num_timesteps)):
            if self.weight_sharing:
                s = self.s[1](z1).nodes
                t = self.t[1](z1).nodes
                #s = self.s[1][i](z1).nodes
                #t = self.s[1][i](z1).nodes
            else:
                s = self.s[1][i](z1).nodes
                t = self.t[1][i](z1).nodes
            if self.use_batch_norm:
                bn = self.bns[1][i]
                z1 = z1.replace(nodes=bn.forward(z1.nodes))
            z0 = z0.replace(nodes=(z0.nodes - t) * tf.exp(-s))

            if self.weight_sharing:
                #s = self.s[0][i](z0).nodes
                #t = self.s[0][i](z0).nodes
                s = self.s[0](z0).nodes
                t = self.t[0](z0).nodes
            else:
                s = self.s[0][i](z0).nodes
                t = self.t[0][i](z0).nodes
            if self.use_batch_norm:
                bn = self.bns[0][i]
                z0 = z0.replace(nodes=bn.forward(z0.nodes))
            z1 = z1.replace(nodes=(z1.nodes - t) * tf.exp(-s))
        return z.replace(nodes=tf.concat([z0.nodes, z1.nodes], axis=1))

    def log_prob(self, x):
        z, log_det_jacobian = self.f(x)
        return tf.reduce_sum(self.prior.log_prob(z)) + log_det_jacobian

    def _build(self, input, inverse=True):
        func = self.f if inverse else self.g
        return func(input)


# Copied from graph_nets/modules.py and modified.
class DMSelfAttention(snt.AbstractModule):
    """Multi-head self-attention module.
  The module is based on the following three papers:
   * A simple neural network module for relational reasoning (RNs):
       https://arxiv.org/abs/1706.01427
   * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
   * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.
  The input to the modules consists of a graph containing values for each node
  and connectivity between them, a tensor containing keys for each node
  and a tensor containing queries for each node.
  The self-attention step consist of updating the node values, with each new
  node value computed in a two step process:
  - Computing the attention weights between each node and all of its senders
   nodes, by calculating sum(sender_key*receiver_query) and using the softmax
   operation on all attention weights for each node.
  - For each receiver node, compute the new node value as the weighted average
   of the values of the sender nodes, according to the attention weights.
  - Nodes with no received edges, get an updated value of 0.
  Values, keys and queries contain a "head" axis to compute independent
  self-attention for each of the heads.
  """

    def __init__(self, kq_dim_division, kq_dim, name="dm_self_attention"):
        """Inits the module.
    Args:
      name: The module name.
    """
        super(DMSelfAttention, self).__init__(name=name)
        self._normalizer = gn.modules._unsorted_segment_softmax
        self._kq_dim_division = kq_dim_division
        self._kq_dim = kq_dim

    def _build(self, node_values, node_keys, node_queries, attention_graph):
        """Connects the multi-head self-attention module.
    The self-attention is only computed according to the connectivity of the
    input graphs, with receiver nodes attending to sender nodes.
    Args:
      node_values: Tensor containing the values associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, key_size].
      node_keys: Tensor containing the key associated to each of the nodes. The
        expected shape is [total_num_nodes, num_heads, key_size].
      node_queries: Tensor containing the query associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, query_size]. The
        query size must be equal to the key size.
      attention_graph: Graph containing connectivity information between nodes
        via the senders and receivers fields. Node A will only attempt to attend
        to Node B if `attention_graph` contains an edge sent by Node A and
        received by Node B.
    Returns:
      An output `graphs.GraphsTuple` with updated nodes containing the
      aggregated attended value for each of the nodes with shape
      [total_num_nodes, num_heads, value_size].
    Raises:
      ValueError: if the input graph does not have edges.
    """

        # Sender nodes put their keys and values in the edges.
        # [total_num_edges, num_heads, query_size]
        sender_keys = gn.blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_keys))
        # [total_num_edges, num_heads, value_size]
        sender_values = gn.blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_values))

        # Receiver nodes put their queries in the edges.
        # [total_num_edges, num_heads, key_size]
        receiver_queries = gn.blocks.broadcast_receiver_nodes_to_edges(
            attention_graph.replace(nodes=node_queries))

        # Attention weight for each edge.
        # [total_num_edges, num_heads]
        attention_weights_logits = tf.reduce_sum(sender_keys *
                                                 receiver_queries,
                                                 axis=-1)
        if self._kq_dim_division:
            attention_weights_logits /= tf.sqrt(
                tf.cast(self._kq_dim, dtype=tf.float32))
        normalized_attention_weights = gn.modules._received_edges_normalizer(
            attention_graph.replace(edges=attention_weights_logits),
            normalizer=self._normalizer)

        # Attending to sender values according to the weights.
        # [total_num_edges, num_heads, embedding_size]
        attended_edges = sender_values * normalized_attention_weights[..., None]

        # Summing all of the attended values from each node.
        # [total_num_nodes, num_heads, embedding_size]
        received_edges_aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
            reducer=tf.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(
            attention_graph.replace(edges=attended_edges))

        return attention_graph.replace(nodes=aggregated_attended_values)


class DMSelfAttentionMLP(snt.AbstractModule):
    def __init__(self,
                 kq_dim,
                 v_dim,
                 make_mlp_fn,
                 num_heads=8,
                 concat_heads_output_dim=20,
                 concat=True,
                 residual=False,
                 layer_norm=False,
                 kq_dim_division=False,
                 name="dm_self_attention"):
        super(DMSelfAttentionMLP, self).__init__(name=name)
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.mlp = make_mlp_fn()
        self.num_heads = num_heads
        self.concat_heads_output_dim = concat_heads_output_dim
        self.concat = concat
        self.residual = residual
        self.layer_norm = layer_norm
        self.kq_dim_division = kq_dim_division

    def _build(self, graph):
        initializers = {
            'w': tf.contrib.layers.xavier_initializer(uniform=True),
        }

        # [batch_size, num_heads * kq_dim].
        project_q_mod = snt.Linear(self.num_heads * self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_q = project_q_mod(graph.nodes)
        project_k_mod = snt.Linear(self.num_heads * self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_k = project_k_mod(graph.nodes)

        # At the end of this block, project_q_mod and project_k_mod are both
        # [batch_size, num_heads, kq_dim].
        project_q = tf.reshape(project_q, [-1, self.num_heads, self.kq_dim])
        project_k = tf.reshape(project_k, [-1, self.num_heads, self.kq_dim])

        # At the end of this block, project_v is [batch_size, num_heads,
        # v_dim].
        project_v_mod = snt.Linear(self.v_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_v = project_v_mod(graph.nodes)
        project_v = tf.keras.backend.repeat(project_v, self.num_heads)

        attn_module = DMSelfAttention(self.kq_dim_division, self.kq_dim)
        attn_graph = attn_module(project_v, project_q, project_k, graph)

        # [batch_size, num_heads, v_dim].
        new_nodes = attn_graph.nodes
        new_nodes = tf.reshape(new_nodes, [-1, self.num_heads * self.v_dim])

        # At this point, new_nodes is [batch_size, num_heads * v_dim].
        new_node_proj = snt.Linear(self.concat_heads_output_dim,
                                   use_bias=False)
        new_nodes = new_node_proj(new_nodes)

        if self.concat:
            new_nodes = tf.concat([graph.nodes, new_nodes], axis=1)
        new_nodes = self.mlp(new_nodes)

        if self.residual:
            new_nodes += graph.nodes

        if self.layer_norm:
            ln_mod = snt.LayerNorm()
            new_nodes = ln_mod(new_nodes)
        return graph.replace(nodes=new_nodes)


def dm_self_attn_gnn(kq_dim,
                     v_dim,
                     make_mlp_fn,
                     num_heads,
                     concat_heads_output_dim,
                     concat=True,
                     residual=False,
                     layer_norm=False,
                     kq_dim_division=False):
    return DMSelfAttentionMLP(kq_dim=kq_dim,
                              v_dim=v_dim,
                              make_mlp_fn=make_mlp_fn,
                              num_heads=num_heads,
                              concat_heads_output_dim=concat_heads_output_dim,
                              concat=concat,
                              residual=residual,
                              layer_norm=layer_norm,
                              kq_dim_division=kq_dim_division)


class MultiheadSelfAttention(snt.AbstractModule):
    def __init__(self,
                 kq_dim,
                 v_dim,
                 concat_heads_output_dim,
                 make_mlp_fn,
                 num_heads=1,
                 kq_dim_division=True,
                 layer_norm=False,
                 name="multihead_self_attention"):
        super(MultiheadSelfAttention, self).__init__(name=name)
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.concat_heads_output_dim = concat_heads_output_dim
        self.mlp = make_mlp_fn()
        self.num_heads = num_heads
        self.kq_dim_division = kq_dim_division
        self.layer_norm = layer_norm

    def _build(self, graph):
        n_node = tf.shape(graph.nodes)[0]
        initializers = {
            'w': tf.contrib.layers.xavier_initializer(uniform=True),
        }

        project_q_mod = snt.Linear(self.num_heads * self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_k_mod = snt.Linear(self.num_heads * self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_v_mod = snt.Linear(self.num_heads * self.v_dim,
                                   use_bias=False,
                                   initializers=initializers)

        # After this block, both have shape [num_heads, n_node, kq_dim].
        project_q = tf.reshape(project_q_mod(graph.nodes),
                               [n_node, self.num_heads, self.kq_dim])
        project_q = tf.transpose(project_q, perm=[1, 0, 2])
        project_k = tf.reshape(project_k_mod(graph.nodes),
                               [n_node, self.num_heads, self.kq_dim])
        project_k = tf.transpose(project_k, perm=[1, 0, 2])

        # Shape is [num_heads, n_node, v_dim].
        project_v = tf.reshape(project_v_mod(graph.nodes),
                               [n_node, self.num_heads, self.v_dim])
        project_v = tf.transpose(project_v, perm=[1, 0, 2])

        # Attention weights. After this block, attn_weights shape is [num_heads, n_node, n_node].
        logits = tf.matmul(project_q, project_k, transpose_b=True)
        if self.kq_dim_division:
            logits /= tf.sqrt(tf.cast(self.kq_dim, tf.float32))
        lm = loss_mask(graph)
        mask = 100000 * (1 - lm)
        logits -= mask
        attn_weights = tf.nn.softmax(logits, axis=-1)

        # used for stability.
        maxes = tf.reduce_max(logits, axis=-1)
        maxes = tf.expand_dims(maxes, 2)
        maxes = tf.tile(maxes, multiples=[1, 1, n_node])
        logits -= maxes

        # Shape is [num_heads, n_node, v_dim].
        attended_nodes = tf.matmul(attn_weights, project_v)

        # Shape is [n_node, num_heads * v_dim].
        attended_nodes = tf.reshape(
            tf.transpose(attended_nodes, perm=[1, 0, 2]),
            [n_node, self.num_heads * self.v_dim])

        # After this block shape is [n_node, concat_heads_output_dim].
        project_multihead_mod = snt.Linear(self.concat_heads_output_dim,
                                           use_bias=False,
                                           initializers=initializers)
        project_multihead = project_multihead_mod(attended_nodes)

        concat_nodes = tf.concat([graph.nodes, project_multihead], axis=-1)
        new_nodes = self.mlp(concat_nodes)

        if self.layer_norm:
            ln_mod = snt.LayerNorm()
            new_nodes = ln_mod(new_nodes)
        return graph.replace(nodes=new_nodes)


def multihead_self_attn_gnn(kq_dim,
                            v_dim,
                            concat_heads_output_dim,
                            make_mlp_fn,
                            num_heads=1,
                            kq_dim_division=True,
                            layer_norm=False):
    return MultiheadSelfAttention(
        kq_dim=kq_dim,
        v_dim=v_dim,
        concat_heads_output_dim=concat_heads_output_dim,
        make_mlp_fn=make_mlp_fn,
        num_heads=num_heads,
        kq_dim_division=kq_dim_division,
        layer_norm=layer_norm)


class SelfAttention(snt.AbstractModule):
    def __init__(self,
                 kq_dim,
                 v_dim,
                 make_mlp_fn,
                 kq_dim_division,
                 name="self_attention"):
        super(SelfAttention, self).__init__(name=name)
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.mlp = make_mlp_fn()
        self.kq_dim_division = kq_dim_division

    def _build(self, graph):
        n_node = tf.shape(graph.nodes)[0]
        initializers = {
            'w': tf.contrib.layers.xavier_initializer(uniform=True),
        }

        project_q_mod = snt.Linear(self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_k_mod = snt.Linear(self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_v_mod = snt.Linear(self.v_dim,
                                   use_bias=False,
                                   initializers=initializers)

        # After this block, shape is [num_nodes, kq_dim].
        project_q = project_q_mod(graph.nodes)
        project_k = project_k_mod(graph.nodes)

        # After this block, shape is [num_nodes, v_dim].
        project_v = project_v_mod(graph.nodes)

        # logits shape is [num_nodes, num_nodes].
        logits = tf.matmul(project_q, tf.transpose(project_k))
        if self.kq_dim_division:
            logits /= tf.sqrt(tf.cast(self.kq_dim, tf.float32))

        lm = loss_mask(graph)
        mask = 100000 * (1 - lm)
        logits -= mask

        # Used for numerical stability. Maybe unnecessary.
        maxes = tf.reduce_max(logits, axis=-1)
        maxes = tf.expand_dims(maxes, 1)
        maxes = tf.tile(maxes, multiples=[1, n_node])
        logits -= maxes

        attn_weights = tf.nn.softmax(logits, axis=-1)

        attended_nodes = tf.matmul(attn_weights, project_v)
        concat_nodes = tf.concat([graph.nodes, attended_nodes], axis=-1)
        new_nodes = self.mlp(concat_nodes)
        return graph.replace(nodes=new_nodes)


def self_attn_gnn(kq_dim, v_dim, make_mlp_fn, kq_dim_division):
    return SelfAttention(kq_dim, v_dim, make_mlp_fn, kq_dim_division)


def loss_mask_padded(graph, max_n_node):
    def body(i, sizes, max_n_node, output):
        n_node = sizes[i]
        g = tf.ones([n_node, n_node])
        g = tf.pad(g, [[0, max_n_node - n_node], [0, max_n_node - n_node]])
        output = output.write(i, g)
        return (i + 1, sizes, max_n_node, output)

    num_graphs = gn.utils_tf.get_num_graphs(graph)
    loop_condition = lambda i, *_: tf.less(i, num_graphs)
    initial_loop_vars = [
        0, graph.n_node, max_n_node,
        tf.TensorArray(dtype=tf.float32, size=num_graphs, infer_shape=False)
    ]
    _, _, _, output = tf.while_loop(loop_condition,
                                       body,
                                       initial_loop_vars,
                                       back_prop=False)
    return output.stack()


class LatestSelfAttention(snt.AbstractModule):
    def __init__(self,
                 kq_dim,
                 v_dim,
                 concat_heads_output_dim,
                 make_mlp_fn,
                 train_batch_size,
                 max_n_node,
                 num_heads=1,
                 kq_dim_division=True,
                 layer_norm=False,
                 name="latest_self_attention"):
        super(LatestSelfAttention, self).__init__(name=name)
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.concat_heads_output_dim = concat_heads_output_dim
        self.mlp = make_mlp_fn()
        self.num_heads = num_heads
        self.kq_dim_division = kq_dim_division
        self.layer_norm = layer_norm
        self.train_batch_size = train_batch_size
        self.max_n_node = max_n_node

        #with self._enter_variable_scope():
        #    self.wk = tf.keras.layers.Dense(num_heads * kq_dim, use_bias=False, name=name + "/keras_wk")
        #    self.wq = tf.keras.layers.Dense(num_heads * kq_dim, use_bias=False, name=name + "/keras_wq")
        #    self.wv = tf.keras.layers.Dense(num_heads * v_dim, use_bias=False, name=name + "/keras_wv")
        #    self.output_weights = tf.keras.layers.Dense(concat_heads_output_dim, use_bias=False, name=name + "/keras_output")


    def pad(self, node_embeddings, n_node):
        n_node_diffs = self.max_n_node - n_node
        graph_list = tf.split(node_embeddings, n_node, 0)
        padded_graphs = []
        padding = tf.constant([[0, 1], [0, 0]])
        for i in range(self.train_batch_size):
            padded_graphs.append(tf.pad(graph_list[i], padding * n_node_diffs[i]))
        final_output = tf.stack(padded_graphs)
        return final_output


    def unpad(self, node_embeddings, n_node):
        graph_list = tf.unstack(node_embeddings, axis=0)
        to_concat = []
        for i in range(self.train_batch_size):
            to_concat.append(graph_list[i][0:n_node[i]])
        final_output = tf.concat(to_concat, axis=0)
        return final_output

    def split_heads(self, node_embeddings, depth):
        node_embeddings = tf.reshape(node_embeddings, (self.train_batch_size, -1, self.num_heads, depth))
        return tf.transpose(node_embeddings, perm=[0, 2, 1, 3])

    def split_heads(self, x, depth):
        """Split the last dimension into (num_heads, depth). 
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (self.train_batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _build(self, graph):
        # graph_batch_size = total number of graphs in the batch,
        # node_batch_size = total number of nodes in all the graphs in the
        # batch, so node_batch_size > graph_batch_size.
        node_embeddings = graph.nodes

        # Shape [node_batch_size, num_heads * kq_dim | v_dim].
        project_k = tf.layers.dense(node_embeddings, self.num_heads * self.kq_dim, use_bias=False)
        project_q = tf.layers.dense(node_embeddings, self.num_heads * self.kq_dim, use_bias=False)
        project_v = tf.layers.dense(node_embeddings, self.num_heads * self.v_dim, use_bias=False)

        # Shape is [graph_batch_size, max_n_node, num_heads * kq_dim | v_dim].
        project_k = self.pad(project_k, graph.n_node)
        project_q = self.pad(project_q, graph.n_node)
        project_v = self.pad(project_v, graph.n_node)

        # After this block, shape is [graph_batch_size, num_heads, max_n_node, kq_dim | v_dim].
        project_k = self.split_heads(project_k, self.kq_dim)
        project_q = self.split_heads(project_q, self.kq_dim)
        project_v = self.split_heads(project_v, self.v_dim)

        # Attention weights. After this block, attn_weights shape is [graph_batch_size, num_heads, n_node, n_node].
        logits = tf.matmul(project_q, project_k, transpose_b=True)
        if self.kq_dim_division:
            logits /= tf.sqrt(tf.cast(self.kq_dim, tf.float32))

        # Mask the logits.
        lm = loss_mask_padded(graph, self.max_n_node)
        tiled_lm = tf.reshape(tf.tile(lm, [1, self.num_heads, 1]), [self.train_batch_size, self.num_heads, self.max_n_node, self.max_n_node])
        logits = logits - 1e9 * (1 - tiled_lm)
        attn_weights = tf.nn.softmax(logits, axis=-1)

        # Shape is [graph_batch_size, num_heads, max_n_node, v_dim].
        attended_nodes = tf.matmul(attn_weights, project_v)
        attended_nodes = tf.transpose(attended_nodes, perm=[0, 2, 1, 3])
        attended_nodes = tf.reshape(attended_nodes, (self.train_batch_size, -1, self.num_heads * self.v_dim))

        # Shape is [node_batch_size, v_dim].
        attended_nodes = self.unpad(attended_nodes, graph.n_node)
        project_multihead = tf.layers.dense(attended_nodes, self.concat_heads_output_dim, use_bias=False)

        concat_nodes = tf.concat([graph.nodes, project_multihead], axis=-1)
        new_nodes = self.mlp(concat_nodes)

        if self.layer_norm:
            ln_mod = snt.LayerNorm()
            new_nodes = ln_mod(new_nodes)

        return graph.replace(nodes=new_nodes)


def latest_self_attention_gnn(kq_dim,
                              v_dim,
                              concat_heads_output_dim,
                              make_mlp_fn,
                              train_batch_size,
                              max_n_node,
                              num_heads=1,
                              kq_dim_division=True,
                              layer_norm=False,
                              name='latest_self_attention'):
    return LatestSelfAttention(
        kq_dim=kq_dim,
        v_dim=v_dim,
        concat_heads_output_dim=concat_heads_output_dim,
        make_mlp_fn=make_mlp_fn,
        train_batch_size=train_batch_size,
        max_n_node=max_n_node,
        num_heads=num_heads,
        kq_dim_division=kq_dim_division,
        layer_norm=layer_norm,
        name=name)


def pairwise_concat(nodes):
    n_nodes = tf.shape(nodes)[0]
    node_embedding_dim = tf.shape(nodes)[1]
    tile_as = tf.reshape(tf.tile(nodes, [1, n_nodes]), [n_nodes * n_nodes, node_embedding_dim])
    tile_as.set_shape([None, 40])
    tile_bs = tf.tile(nodes, [n_nodes, 1])
    toret = tf.concat([tile_as, tile_bs], axis=-1)
    return toret


class AdditiveSelfAttention(snt.AbstractModule):
    def __init__(self,
                 v_dim,
                 attn_mlp_fn,
                 attn_output_dim,
                 gnn_mlp_fn,
                 scaling=True,
                 name="additive_self_attention"):
        super(AdditiveSelfAttention, self).__init__(name=name)
        self.v_dim = v_dim
        self.attn_mlp = attn_mlp_fn()
        self.attn_output_dim = attn_output_dim
        self.gnn_mlp = gnn_mlp_fn()
        self.scaling = scaling

    def _build(self, graph):
        initializers = {
            'w': tf.contrib.layers.xavier_initializer(uniform=True),
        }
        node_embedding_dim = tf.cast(tf.shape(graph.nodes)[1], dtype=tf.float32)
        project_v_mod = snt.Linear(self.v_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_v = project_v_mod(graph.nodes)
        n_node = tf.shape(graph.nodes)[0]
        pairwise_nodes = pairwise_concat(graph.nodes)
        output = self.attn_mlp(pairwise_nodes)

        dot_var = tf.get_variable('dot_var',
                                  shape=self.attn_output_dim,
                                  dtype=tf.float32,
                                  initializer=tf.glorot_normal_initializer,
                                  trainable=True)
        d = tf.tensordot(dot_var, output, axes=[0, 1])
        logits = tf.reshape(d, [n_node, n_node])

        # logits shape is [num_nodes, num_nodes].
        if self.scaling:
            logits /= tf.sqrt(node_embedding_dim)

        lm = loss_mask(graph)
        mask = 100000 * (1 - lm)
        logits -= mask

        # Used for numerical stability. Maybe unnecessary.
        maxes = tf.reduce_max(logits, axis=-1)
        maxes = tf.expand_dims(maxes, 1)
        maxes = tf.tile(maxes, multiples=[1, n_node])
        logits -= maxes

        attn_weights = tf.nn.softmax(logits, axis=-1)

        attended_nodes = tf.matmul(attn_weights, project_v)
        concat_nodes = tf.concat([graph.nodes, attended_nodes], axis=-1)
        new_nodes = self.gnn_mlp(concat_nodes)
        return graph.replace(nodes=new_nodes)


def additive_self_attn_gnn(v_dim,
                           attn_mlp_fn,
                           attn_output_dim,
                           gnn_mlp_fn):
    return AdditiveSelfAttention(
        v_dim=v_dim,
        attn_mlp_fn=attn_mlp_fn,
        attn_output_dim=attn_output_dim,
        gnn_mlp_fn=gnn_mlp_fn)


def pairwise_concat_padded(nodes):
    n_nodes = tf.shape(nodes)[1]
    n_graphs = tf.shape(nodes)[0]
    tile_as = tf.tile(nodes, [1, 1, n_nodes])
    tile_as = tf.reshape(tile_as, [n_graphs, n_nodes**2, -1])
    tile_bs = tf.tile(nodes, [1, n_nodes, 1])
    return tf.concat([tile_as, tile_bs], axis=-1)


class PaddedAdditiveSelfAttention(snt.AbstractModule):
    def __init__(self,
                 v_dim,
                 attn_mlp_fn,
                 attn_output_dim,
                 gnn_mlp_fn,
                 max_n_node,
                 train_batch_size,
                 node_embedding_dim,
                 scaling=True,
                 name="padded_additive_self_attention"):
        super(PaddedAdditiveSelfAttention, self).__init__(name=name)
        self.v_dim = v_dim
        self.attn_mlp = attn_mlp_fn()
        self.gnn_mlp = gnn_mlp_fn()
        self.attn_output_dim = attn_output_dim
        self.max_n_node = max_n_node
        self.scaling=scaling
        self.train_batch_size = train_batch_size
        self.node_embedding_dim = node_embedding_dim
        self.wv = tf.keras.layers.Dense(v_dim, use_bias=False)


    def pad(self, node_embeddings, n_node):
        n_node_diffs = self.max_n_node - n_node
        graph_list = tf.split(node_embeddings, n_node, 0)
        padded_graphs = []
        padding = tf.constant([[0, 1], [0, 0]])
        for i in range(self.train_batch_size):
            padded_graphs.append(tf.pad(graph_list[i], padding * n_node_diffs[i]))
        final_output = tf.stack(padded_graphs)
        return final_output


    def unpad(self, node_embeddings, n_node):
        graph_list = tf.unstack(node_embeddings, axis=0)
        to_concat = []
        for i in range(self.train_batch_size):
            to_concat.append(graph_list[i][0:n_node[i]])
        final_output = tf.concat(to_concat, axis=0)
        return final_output

    def _build(self, graph):
        # train_batch_size = total number of graphs in the batch,
        # node_batch_size = total number of nodes in all the graphs in the
        # batch, so node_batch_size > graph_batch_size.
        node_embeddings = graph.nodes
        node_embeddings = tf.Print(node_embeddings, [node_embeddings], "node_embeddings is: ", summarize=100, first_n=1)

        # [node_batch_size, v_dim].
        project_v = self.wv(node_embeddings)
        project_v = tf.Print(project_v, [project_v], "project v is: ", summarize=100, first_n=1)
        padded_project_v = self.pad(project_v, graph.n_node)
        padded_project_v.set_shape([self.train_batch_size, self.max_n_node, self.v_dim])
        padded_project_v = tf.Print(padded_project_v, [padded_project_v], "padded project v is ", summarize=100, first_n=1)

        # [train_batch_size, max_n_node, node_embedding_dim].
        padded_node_embeddings = self.pad(node_embeddings, graph.n_node)
        padded_node_embeddings = tf.Print(padded_node_embeddings, [padded_node_embeddings], "padded_node_embeddings is: ", summarize=100, first_n=1)

        # [train_batch_size, max_n_node * max_n_node, node_embedding_dim * 2].
        pairwise_concat = pairwise_concat_padded(padded_node_embeddings)

        # [graph_batch_size * max_n_node * max_n_node, node_embedding_dim * 2].
        pairwise_concat = tf.reshape(pairwise_concat, [self.train_batch_size * self.max_n_node ** 2, self.node_embedding_dim])
        pairwise_concat.set_shape([None, self.node_embedding_dim])

        # [graph_batch_size * max_n_node * max_n_node, attn_output_dim].
        additive_attn_output = self.attn_mlp(pairwise_concat)
        dot_var = tf.get_variable('dot_var',
                                  shape=[self.attn_output_dim],
                                  dtype=tf.float32,
                                  initializer=tf.glorot_normal_initializer,
                                  trainable=True)
        d = tf.tensordot(dot_var, additive_attn_output, axes=[0, 1])
        logits = tf.reshape(d, [self.train_batch_size, self.max_n_node, self.max_n_node])
        logits.set_shape([self.train_batch_size, self.max_n_node, self.max_n_node])
        lm = loss_mask_padded(graph, self.max_n_node)
        logits = logits - 100000 * (1 - lm)

        # logits shape is [num_nodes, num_nodes].
        #if self.scaling:
            #logits /= tf.sqrt(tf.cast(node_embedding_dim, dtype=tf.float32))
        attn_weights = tf.nn.softmax(logits, axis=-1)

        attended_nodes = tf.matmul(attn_weights, padded_project_v)
        attended_nodes = self.unpad(attended_nodes, graph.n_node)
        concat_nodes = tf.concat([graph.nodes, attended_nodes], axis=-1)
        #concat_nodes.set_shape([None, self.node_embedding_dim / 2 + self.attn_output_dim])
        new_nodes = self.gnn_mlp(concat_nodes)
        return graph.replace(nodes=new_nodes)


def padded_additive_self_attn_gnn(v_dim,
                                  attn_mlp_fn,
                                  attn_output_dim,
                                  gnn_mlp_fn,
                                  max_n_node,
                                  train_batch_size,
                                  node_embedding_dim):
        return PaddedAdditiveSelfAttention(
            v_dim=v_dim,
            attn_mlp_fn=attn_mlp_fn,
            attn_output_dim=attn_output_dim,
            gnn_mlp_fn=gnn_mlp_fn,
            max_n_node=max_n_node,
            train_batch_size=train_batch_size,
            node_embedding_dim=node_embedding_dim)
