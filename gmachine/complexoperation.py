import networkx as nx
from random import choice
import random
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
import json
from itertools import combinations


# Operation 1

def add_node(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()

    if len(new_graph.nodes) < 1:
        new_graph.add_node(0)
    else:
        v_new = max(new_graph.nodes) + 1
        new_graph.add_node(v_new)
    return new_graph


# Inverse Add Node
def inv_add_node(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    removed_node_connections = []
    removed_nodes = []

    degree_view = nx.degree(new_graph)
    deg_view_dict = {key: value for key, value in degree_view}
    removed_node = min_deg_node = min(deg_view_dict, key=deg_view_dict.get)

    neighbor_nodes = new_graph.neighbors(min_deg_node)
    # get list of nodes connected to high degree centrality
    removed_node_neighbors = [n for n in neighbor_nodes]
    nnl_size = len(removed_node_neighbors)
    removed_node_connections.extend(zip([removed_node] * nnl_size, removed_node_neighbors))
    removed_nodes.extend([removed_node])
    new_graph.remove_node(min_deg_node)
    return new_graph, removed_nodes, removed_node_connections  # min_deg_node is the removed node


# Operation 2

def add_one_node_rand_edge(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    if len(new_graph.nodes) < 1:
        raise Exception("Sorry, Input graph has number of nodes less than one")

    node_list = [n for n in new_graph.nodes()]
    v_random = random.choice(node_list)
    v_new = max(new_graph.nodes) + 1
    new_graph.add_node(v_new)
    new_graph.add_edge(v_new, v_random)
    return new_graph


def inv_add_one_node_rand_edge(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    gcc = sorted(nx.connected_components(new_graph), key=len, reverse=True)
    if len(gcc) > 1:
        raise Exception("Sorry, Input graph has unconnected subgraphs")

    if len(new_graph.nodes) < 2:
        raise Exception("Sorry, Input graph has number of nodes less than one")

    node_list = [n for n in new_graph.nodes()]
    removed_node = v_random = random.choice(node_list)

    neighbor_nodes = new_graph.neighbors(removed_node)
    removed_node_neighbors = [n for n in neighbor_nodes]

    new_graph.remove_node(v_random)
    return new_graph, removed_node, removed_node_neighbors


# Operation 3

# n_node = 3
# n_edges = 3
def add_n_nodes_n_edges(input_graph: nx.Graph, n_node=3, n_edges=3):
    assert input_graph is not None
    new_graph = input_graph.copy()
    if len(new_graph.nodes) < 1:
        raise Exception("Sorry, Input graph has number of nodes less than one")

    # node list
    node_list = [n for n in new_graph.nodes()]
    # get max node
    v_max = max(new_graph.nodes)
    # add n nodes
    v_new = []
    for n in range(n_node):
        v_new.append(v_max + 1 + n)

    for e in range(n_edges):
        for v in v_new:
            graph_rand_v = random.choice(node_list)
            if not new_graph.has_edge(v, graph_rand_v):
                new_graph.add_edge(v, graph_rand_v)
    return new_graph


# n_node = 3
def inv_add_n_nodes_n_edges(input_graph: nx.Graph, n_node=3):
    assert input_graph is not None
    new_graph = input_graph.copy()
    if len(new_graph.nodes) < n_node:
        raise Exception("Sorry, Input graph has number of nodes less than one")

    removed_nodes = []
    removed_node_connections = []

    # node list

    for n in range(n_node):
        dgx = new_graph.copy()
        degree_view = nx.degree(dgx)
        deg_view_dict = {key: value for key, value in degree_view}
        v_rand_choice = random.choice(list(deg_view_dict.keys()))
        dgx.remove_node(v_rand_choice)
        gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
        if (len(gcc) == 1):
            neighbor_nodes = new_graph.neighbors(v_rand_choice)
            new_graph.remove_node(v_rand_choice)
            removed_node_neighbors = [n for n in neighbor_nodes]
            nnl_size = len(removed_node_neighbors)
            removed_node_connections.extend(zip([v_rand_choice] * nnl_size, removed_node_neighbors))
            removed_nodes.extend([v_rand_choice])

        elif not len(gcc) == 1:
            removed_node = min(deg_view_dict, key=deg_view_dict.get)
            neighbor_nodes = new_graph.neighbors(removed_node)
            removed_node_neighbors = [n for n in neighbor_nodes]
            nnl_size = len(removed_node_neighbors)
            removed_node_connections.extend(zip([removed_node] * nnl_size, removed_node_neighbors))
            removed_nodes.extend([removed_node])
            new_graph.remove_node(removed_node)

    return new_graph, removed_nodes, removed_node_connections


# operation 4

# triangle_n_edges = 5
def add_triangle_n_edges(input_graph: nx.Graph, triangle_n_edges=5):
    assert input_graph is not None
    new_graph = input_graph.copy()
    if len(new_graph.nodes) < 1:
        raise Exception("Sorry, Input graph has number of nodes less than one")

    # node list
    node_list = [n for n in new_graph.nodes()]
    # create a triangle
    v_max = max(new_graph.nodes)
    # create a triangle
    new_graph.add_edge(v_max + 1, v_max + 2)
    new_graph.add_edge(v_max + 2, v_max + 3)
    new_graph.add_edge(v_max + 1, v_max + 3)

    v_news = [v_max + 1, v_max + 2, v_max + 3]
    for i in range(triangle_n_edges):
        tri_rand_v = random.choice(v_news)
        graph_rand_v = random.choice(node_list)
        if not new_graph.has_edge(tri_rand_v, graph_rand_v):
            new_graph.add_edge(tri_rand_v, graph_rand_v)
    return new_graph


# check to see if there is any valid triangle that can be deleted.

def has_triangle(G, tri_node):
    triangle_node = []
    is_in_triangle = False
    for node in tri_node:
        for nbr1, nbr2 in combinations(G.neighbors(node), 2):
            if G.has_edge(nbr1, nbr2):
                dgx = G.copy()
                dgx.remove_node(nbr1)
                dgx.remove_node(node)
                dgx.remove_node(nbr2)
                gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                if len(gcc) == 1:
                    is_in_triangle = True
                    triangle_node.extend([nbr1, node, nbr2])
                    return is_in_triangle, triangle_node
    return is_in_triangle, triangle_node


def inv_add_triangle_n_edges(input_graph: nx.Graph):
    assert input_graph is not None

    new_graph = input_graph.copy()
    if len(new_graph.nodes) < 3:
        raise Exception("Sorry, Input graph has number of nodes less than three")

    removed_nodes = []
    removed_node_connections = []

    # node list
    triangle_nodes = nx.triangles(new_graph)
    tri_node = [key for key, value in triangle_nodes.items() if value >= 1]
    if len(tri_node) == 0:
        raise Exception("Sorry, there are no triangles in the graph")
    is_in_triangle = False
    have_triangle, triangle_node = has_triangle(input_graph, tri_node)
    if have_triangle:
        removed_nodes.extend(triangle_node)
        for node in triangle_node:
            neighbor_nodes = new_graph.neighbors(node)
            removed_node_neighbors = [n for n in neighbor_nodes]
            nnl_size = len(removed_node_neighbors)
            removed_node_connections.extend(zip([node] * nnl_size, removed_node_neighbors))
            new_graph.remove_node(node)

    return new_graph, removed_nodes, removed_node_connections


def stop():
    pass


class ComplexActions:
    add_node = 0
    add_one_node_rand_edge = 1
    add_triangle_n_edges = 2
    add_n_nodes_n_edges = 3
    stop = 4


def construction_sequence_to_graph(sequence, start_node_from: int = 0):
    g = nx.Graph()
    action_idx = 0
    while action_idx < len(sequence):
        action = sequence[action_idx]
        if ComplexActions.add_node == action:
            g = add_node(g)
            action_idx += 1
        elif ComplexActions.add_one_node_rand_edge == action:
            g = add_one_node_rand_edge(g)
            action_idx += 1
        elif ComplexActions.add_n_nodes_n_edges == action:
            g = add_n_nodes_n_edges(g)
            action_idx += 1
        elif ComplexActions.add_triangle_n_edges == action:
            g = add_triangle_n_edges(g)
            action_idx += 1
        elif ComplexActions.stop == action:
            break
        else:
            raise ValueError('Sequence error (unknown action) on action idx #{idx}: a={action}'.format(idx=action_idx,
                                                                                                       action=action))
    return g


complex_construction_actions = [
    add_node,
    add_one_node_rand_edge,
    add_triangle_n_edges,
    add_n_nodes_n_edges,
    stop
]


def random_transition_model_for_complex_operation(n, complex_construction_actions, ComplexActions):
    construction_sequence = []
    g = nx.Graph()
    action = ComplexActions.add_node
    g.add_node(len(g.nodes))
    nodes = list(range(len(g.nodes)))
    i = len(g.nodes)
    while i < n:
        if int(ComplexActions.add_node) == int(action):
            g = add_node(g)
            action = random.choice(complex_construction_actions)
            nodes = list(range(len(g.nodes)))
            i = len(nodes)
            construction_sequence.append(ComplexActions.add_node)

        elif ComplexActions.add_one_node_rand_edge == int(action):
            g = add_one_node_rand_edge(g)
            action = random.choice(complex_construction_actions)
            nodes = list(range(len(g.nodes)))
            i = len(nodes)
            construction_sequence.append(ComplexActions.add_one_node_rand_edge)

        elif ComplexActions.add_n_nodes_n_edges == int(action):
            g = add_n_nodes_n_edges(g)
            action = random.choice(complex_construction_actions)
            nodes = list(range(len(g.nodes)))
            i = len(nodes)
            construction_sequence.append(ComplexActions.add_n_nodes_n_edges)

        elif ComplexActions.add_triangle_n_edges == int(action):
            g = add_triangle_n_edges(g)
            action = random.choice(complex_construction_actions)
            nodes = list(range(len(g.nodes)))
            i = len(nodes)
            construction_sequence.append(ComplexActions.add_triangle_n_edges)

        elif ComplexActions.stop == int(action):
            action = random.choice(complex_construction_actions)
            construction_sequence.append(ComplexActions.stop)
    return g, construction_sequence


def generated_operation_count_per_loc(construction_sequence: dict, operations: list):
    seq_size = len(construction_sequence['construction_sequence'])
    sequence_lengths = []
    max_seq = 0
    for i in range(seq_size):
        seq = construction_sequence['construction_sequence'][i]
        len_seq = len(seq)
        sequence_lengths.append(len_seq)
    max_seq = max(sequence_lengths)

    new_construction_sequence = []
    loc = 0
    for i in range(seq_size):
        seq = construction_sequence['construction_sequence'][i]
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


# Deconstruction of graph functionalities

class DeconstructionAction:
    inv_add_node = 0
    inv_add_one_node_rand_edge = 1
    inv_add_triangle_n_edges = 2
    inv_add_n_nodes_n_edges = 3


inv_actions = [
    DeconstructionAction.inv_add_node,
    DeconstructionAction.inv_add_one_node_rand_edge,
    DeconstructionAction.inv_add_triangle_n_edges,
    DeconstructionAction.inv_add_n_nodes_n_edges,
]

inverse_actions = [
    inv_add_node,
    inv_add_one_node_rand_edge,
    inv_add_triangle_n_edges,
    inv_add_n_nodes_n_edges,
]


def random_deconstruction_model(graph: nx.Graph, inverse_action, m=None, seed=None):
    deconstruction_sequence = []
    g = graph.copy()
    i = 0
    n = len(g.nodes)
    n_node = 3
    # plt.subplot(1, 2, 1)
    # nx.draw(g, with_labels=True)
    # plt.show()

    while i < n - 1:
        valid_random_actions = []
        # apply inverse operations until valid actions exists
        # i.e. if no valid action is found then, back-jump one step backward and re-apply actions
        while not valid_random_actions:
            for act in range(len(inverse_action)):  # take 1,2,3 values from inverse actions
                if DeconstructionAction.inv_add_node == act and len(g.nodes) >= 1:
                    dgx, removed_node, removed_node_neighbors = inverse_action[act](g)
                    gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                    if len(gcc) == 1:
                        valid_random_actions.append(act)

                if DeconstructionAction.inv_add_one_node_rand_edge == act and len(g.nodes) >= 1:
                    dgx, removed_node, removed_node_neighbors = inverse_action[act](g)
                    gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                    if len(gcc) == 1:
                        valid_random_actions.append(act)

                triangle_nodes = nx.triangles(g)
                tri_node = [key for key, value in triangle_nodes.items() if value >= 1]
                if DeconstructionAction.inv_add_triangle_n_edges == act and len(tri_node) > 0:
                    dgx, removed_node, removed_node_neighbors = inverse_action[act](g)
                    gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                    if len(gcc) == 1:
                        valid_random_actions.append(act)

                if DeconstructionAction.inv_add_n_nodes_n_edges == act and len(g.nodes) >= n_node:
                    dgx, removed_node, removed_node_neighbors = inverse_action[act](g)
                    gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                    if len(gcc) == 1:
                        valid_random_actions.append(act)
                if len(g.nodes) == 1:
                    dgx, removed_node, removed_node_neighbors = inverse_action[act](g)
                    valid_random_actions.append(act)

        # print("valid_random_actions", valid_random_actions)
        #
        # Once the valid actions are in the list
        # select one valid action and apply to the graph
        # check if the output graph is again valid, i.e. does not have sub graphs
        # if graph is valid, go to next step else do it again until you get valid output graph
        valid_gcc = False

        while not valid_gcc:
            action = random.choice(valid_random_actions)
            if DeconstructionAction.inv_add_node == action and len(g.nodes) >= 2:
                node_num = len(g.nodes) - 1
                dgx, removed_node, removed_node_neighbors = inverse_action[action](g)
                gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                if len(gcc) == 1:
                    g = dgx.copy()
                    valid_gcc = True

            if DeconstructionAction.inv_add_one_node_rand_edge == action and len(g.nodes) >= 1:
                node_num = len(g.nodes) - 1
                dgx, removed_node, removed_node_neighbors = inverse_action[action](g)
                gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                if len(gcc) == 1:
                    g = dgx.copy()
                    valid_gcc = True

            triangle_nodes = nx.triangles(g)
            tri_node = [key for key, value in triangle_nodes.items() if value >= 1]
            if DeconstructionAction.inv_add_triangle_n_edges == action and len(tri_node) > 0:
                node_num = len(g.nodes) - 1
                dgx, removed_node, removed_node_neighbors = inverse_action[action](g)
                gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                if len(gcc) == 1:
                    g = dgx.copy()
                    valid_gcc = True

            if DeconstructionAction.inv_add_n_nodes_n_edges == action and len(g.nodes) >= n_node:
                node_num = len(g.nodes) - 1
                dgx, removed_node, removed_node_neighbors = inverse_action[action](g)
                gcc = sorted(nx.connected_components(dgx), key=len, reverse=True)
                if len(gcc) == 1:
                    g = dgx.copy()
                    valid_gcc = True

        deconstruction_sequence.append(action)

        if len(g.nodes) == 1:
            g, removed_node, removed_node_neighbors = inverse_action[0](g)
            deconstruction_sequence.append(0)
            # plott(g, removed_node, removed_node_neighbors)
            break

        # plott(g, removed_node, removed_node_neighbors)
        i = i + 1
    return deconstruction_sequence


def generate_deconstruction_sequence(graph: nx.Graph, inverse_action, n_sequence: int):
    deconstruction_sequence = []
    for i in range(n_sequence):
        sequence = random_deconstruction_model(graph, inverse_action)
        deconstruction_sequence.append(sequence)
    return deconstruction_sequence


# Convert deconstruction sequence to construction sequence
class ConstructionAction:
    add_node = 0
    add_one_node_rand_edge = 1
    add_triangle_n_edges = 2
    add_n_nodes_n_edges = 3
    stop = 4


construct_actions = [
    ConstructionAction.add_node,
    ConstructionAction.add_one_node_rand_edge,
    ConstructionAction.add_triangle_n_edges,
    ConstructionAction.add_n_nodes_n_edges,
    ConstructionAction.stop
]

construction_actions = [
    add_node,
    add_one_node_rand_edge,
    add_triangle_n_edges,
    add_n_nodes_n_edges,
    stop
]


def deconstruction_to_construction_sequence(deconstruction_sequence: list):
    construction_sequence = deconstruction_sequence.copy()
    construction_sequence.reverse()
    construction_sequence.append(ConstructionAction.stop)  # add stop to the construction sequence
    return construction_sequence


# Generate construction seqeunce for all datasets

def generate_construction_sequence(path, inverse_action):
    construction_sequences = {}
    graphml_files = os.listdir(path)
    for file in tqdm(graphml_files):
        if file.endswith('.graphml'):
            filepath = path + file
            graph = nx.read_graphml(filepath)
            num_nodes = len(graph.nodes)
            dsequence = generate_deconstruction_sequence(graph, inverse_action, 1)
            csequence = deconstruction_to_construction_sequence(dsequence[0])
            construction_sequences.setdefault(num_nodes, []).append(csequence)
    return construction_sequences
