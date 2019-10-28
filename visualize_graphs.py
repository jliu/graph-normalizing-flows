import os
import pickle
import random
import sys
import utils

from absl import flags
import matplotlib
import networkx as nx
matplotlib.use('agg')
import matplotlib.pyplot as plt

flags.DEFINE_string('graphs_file', '', '')
flags.DEFINE_string('output_dir', '', '')
flags.DEFINE_integer('sample_size', 0, '')
flags.DEFINE_bool('preprocess_graphs', False, '')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def preprocess_graph(G):
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        if weight == 1:
            edge[2]['color'] = 'k'
        # False positives.
        elif weight == 2:
            edge[2]['color'] = 'r'
        # False negatives.
        elif weight == 3:
            edge[2]['color'] = '#d260ff'


def visualize_graph(G, filename=None):
    pos = nx.spring_layout(G, k=0.5, iterations=200, scale=100.0)
    #use for grid graphs
    #pos = nx.spectral_layout(G)

    plt.figure(figsize=(30, 10))

    edge_color = [G[u][v]['color'] for u, v in G.edges()] if FLAGS.preprocess_graphs else 'black'
    nx.draw(G,
            pos,
            node_color='#e4dce8',
            node_size=50,
            with_labels=False,
            width=2.0,
            edge_color=edge_color)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


graphs = pickle.load(open(FLAGS.graphs_file, 'rb'))
graphs.sort(key=lambda x: x.number_of_nodes())
if FLAGS.sample_size > 0:
    graphs = random.sample(graphs, FLAGS.sample_size)

for i in range(len(graphs)):
    print("processing graph {}".format(i))
    g = graphs[i]
    filename = os.path.join(FLAGS.output_dir, 'graph_{}_num_nodes_{}'.format(i, g.number_of_nodes()))
    if FLAGS.preprocess_graphs:
        preprocess_graph(g)
    visualize_graph(graphs[i], filename)
