import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.utils import py_random_state

from gmachine.complexoperation import add_one_node_rand_edge, add_node, add_triangle_n_edges, add_n_nodes_n_edges, stop


class ConstructionAction:
    add_node = 0
    add_edge = 1
    stop = 2


def _random_subset(seq, m, rng):
    targets = set()
    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets


def generate_ba_model_construction_sequence(n, m, seed=None):
    if m < 1 or m >= n:
        raise nx.NetworkXError(f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}")

    # Construction sequence
    sequence = []

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)
    sequence.extend([ConstructionAction.add_node] * m)  # create nodes m times and don't create edges

    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        sequence.append(ConstructionAction.add_node)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        for t in targets:
            sequence.append(ConstructionAction.add_edge)
            sequence.append(source)
            sequence.append(t)

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1

    return sequence


def generate_ba_model_construction_sequence_dataset(
        graph_size_min: int,
        graph_size_max: int,
        ba_model_m_min: int,
        ba_model_m_max: int,
        n_samples,
        fname):
    samples = []
    for _ in range(n_samples):
        size = random.randint(graph_size_min, graph_size_max)
        ba_model_m = random.randint(ba_model_m_min, ba_model_m_max)
        samples.append(generate_ba_model_construction_sequence(size, ba_model_m))

    with open(fname, 'wb') as f:
        pickle.dump(samples, f)

    return samples


def construction_sequence_to_graph(sequence, start_node_from: int = 0):
    g = nx.Graph()
    action_idx = 0
    while action_idx < len(sequence):
        action = sequence[action_idx]
        if ConstructionAction.add_node == action:
            g.add_node(start_node_from + len(g.nodes))
            action_idx += 1
        elif ConstructionAction.add_edge == action:
            action_idx += 1
            source = start_node_from + sequence[action_idx]
            action_idx += 1
            target = start_node_from + sequence[action_idx]
            g.add_edge(source, target)
            action_idx += 1
        else:
            raise ValueError('Sequence error (unknown action) on action idx #{idx}: a={action}'.format(idx=action_idx,
                                                                                                       action=action))
    return g


# Datasets for evaluation of the trained model

# n: number of nodes
# m:

actions = [
    ConstructionAction.add_node,
    ConstructionAction.add_edge,
    ConstructionAction.stop
]


def random_transition_model(n, actions, m=None, seed=None):
    g = nx.Graph()
    action = ConstructionAction.add_node
    g.add_node(len(g.nodes))
    nodes = list(range(len(g.nodes)))
    i = len(g.nodes)
    while i < n:
        if ConstructionAction.add_node == int(action):
            g.add_node(len(g.nodes))
            i += 1
            action = random.choice(actions)
            nodes = list(range(len(g.nodes)))

        if ConstructionAction.add_edge == int(action):
            src_node = random.choice(nodes)
            if m is None:
                dst_node = random.choice(nodes)
                if not g.has_edge(src_node, dst_node):
                    g.add_edge(src_node, dst_node)
            else:
                dst_node = random.sample(nodes, m)
                for d_node in dst_node:
                    if not g.has_edge(src_node, d_node):
                        g.add_edge(src_node, d_node)
            action = random.choice(actions)

        if ConstructionAction.stop == int(action) and len(g.nodes) < n:
            action = ConstructionAction.add_node
    return g


class ComplexConstructionAction:
    add_node = 0
    add_one_node_rand_edge = 1
    add_triangle_n_edges = 2
    add_n_nodes_n_edges = 3
    stop = 4


complex_construction_actions = [
    add_node,
    add_one_node_rand_edge,
    add_triangle_n_edges,
    add_n_nodes_n_edges,
    stop
]


def random_transition_model_for_complex_operation(n, complex_construction_actions, m=None, seed=None):
    g = nx.Graph()
    action = ConstructionAction.add_node
    g.add_node(len(g.nodes))
    nodes = list(range(len(g.nodes)))
    i = len(g.nodes)
    while i < n:
        if ConstructionAction.add_node == int(action):
            g.add_node(len(g.nodes))
            i += 1
            action = random.choice(actions)
            nodes = list(range(len(g.nodes)))

        if ConstructionAction.add_edge == int(action):
            src_node = random.choice(nodes)
            if m is None:
                dst_node = random.choice(nodes)
                if not g.has_edge(src_node, dst_node):
                    g.add_edge(src_node, dst_node)
            else:
                dst_node = random.sample(nodes, m)
                for d_node in dst_node:
                    if not g.has_edge(src_node, d_node):
                        g.add_edge(src_node, d_node)
            action = random.choice(actions)

        if ConstructionAction.stop == int(action) and len(g.nodes) < n:
            action = ConstructionAction.add_node
    return g


def operation_count(construction_sequence, operations, n):
    seq_size = len(construction_sequence['construction_sequence'][str(n)])
    operation_counts = [0] * len(operations)

    for i in range(seq_size):
        seq = construction_sequence['construction_sequence'][str(n)][i]
        for op in range(len(operations)):
            counts = seq.count(op)
            operation_counts[op] = operation_counts[op] + counts
    return operation_counts


def operation_count_per_loc(construction_sequence, operations, n):
    seq_size = len(construction_sequence['construction_sequence'][str(n)])
    sequence_lengths = []
    max_seq = 0
    for i in range(seq_size):
        seq = construction_sequence['construction_sequence'][str(n)][i]
        len_seq = len(seq)
        sequence_lengths.append(len_seq)
    max_seq = max(sequence_lengths)

    new_construction_sequence = []
    loc = 0
    for i in range(seq_size):
        seq = construction_sequence['construction_sequence'][str(n)][i]
        len_seq = len(seq)
        fill_length = max_seq - len_seq
        fill_arr = [np.nan] * fill_length
        seq.extend(fill_arr)
        new_construction_sequence.append(seq)

    new_construction_sequence = np.array(new_construction_sequence)
    new_construction_sequence_transpose = new_construction_sequence.transpose()

    operation_counts_per_loc = []
    for i in range(max_seq):
        operation_counts = [0] * len(operations)
        seq = list(new_construction_sequence_transpose[i])
        for op in range(len(operations)):
            counts = seq.count(op)
            operation_counts[op] = operation_counts[op] + counts
        operation_counts_per_loc.append(operation_counts)

    return operation_counts_per_loc, max_seq


def get_degree_distribution(graphs):
    combine_degree = []
    all_degree = []
    for graph in graphs:
        # clean the graph
        GCC = sorted(nx.connected_components(graph), key=len, reverse=True)
        gg = graph.subgraph(GCC[0])

        degrees = [gg.degree(n) for n in gg.nodes()]
        combine_degree.append(degrees)
        all_degree.extend(degrees)
    return combine_degree, all_degree


def plot_operation_count(operation_counts_A, operation_counts_B, label):
    # create data
    x = ['A', 'B']
    y1 = np.array([operation_counts_A[0], operation_counts_B[0]])
    y2 = np.array([operation_counts_A[1], operation_counts_B[1]])
    y3 = np.array([operation_counts_A[2], operation_counts_B[2]])
    y4 = np.array([operation_counts_A[3], operation_counts_B[3]])

    # plot bars in stack manner
    graph1 = plt.bar(x, y1, color='r')
    graph2 = plt.bar(x, y2, bottom=y1, color='b')
    graph3 = plt.bar(x, y3, bottom=y1 + y2, color='y')
    graph4 = plt.bar(x, y4, bottom=y1 + y2 + y3, color='g')
    plt.xlabel(label)
    plt.ylabel("count")
    plt.legend(["0", "1", "2", "3"])
    plt.title("Operation count in Each dataset")

    plt.show()


def plot_operation_dist_per_loc(operation_counts_per_loc, max_seq, titles):
    # create data
    x = list(range(0, max_seq))
    x = [str(x) for x in x]

    op0 = operation_counts_per_loc[0]
    op1 = operation_counts_per_loc[1]
    op2 = operation_counts_per_loc[2]
    op3 = operation_counts_per_loc[3]

    y1 = np.array(op0)
    y2 = np.array(op1)
    y3 = np.array(op2)
    y4 = np.array(op3)

    # plot bars in stack manner
    plt.subplots(figsize=(20, 2))
    plt.bar(x, y1, color='r')
    plt.bar(x, y2, bottom=y1, color='b')
    plt.bar(x, y3, bottom=y1 + y2, color='y')
    plt.bar(x, y4, bottom=y1 + y2 + y3, color='g')
    plt.xlabel("Sequence Length")
    plt.ylabel("Operation Counts")
    plt.legend(["op-0", "op-1", "op-2", "op-3"])
    plt.title(titles)
    plt.show()
