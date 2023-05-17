from random import choice

import networkx as nx
import numpy as np


def grow(graph: nx.Graph):
    assert graph is not None
    v_new_idx = max(graph.nodes) + 1
    graph.add_node(v_new_idx)
    return graph


def grow_random_single_edge(graph: nx.Graph):
    assert graph is not None
    v_random_neighbor = np.random.choice(list(graph.nodes))
    v_new_idx = max(graph.nodes) + 1
    graph.add_node(v_new_idx)
    graph.add_edge(v_new_idx, v_random_neighbor)
    return graph


def grow_random_edges(graph: nx.Graph, k: int = 0):
    if k < 1:
        k = np.random.randint(1, len(graph.nodes) + 1)
    assert k < len(graph.nodes) + 1
    vs_random_neighbor = np.random.choice(list(graph.nodes), k, replace=False)
    v_new_idx = max(graph.nodes) + 1
    graph.add_node(v_new_idx)
    graph.add_edges_from([(v_new_idx, v_neighbor) for v_neighbor in vs_random_neighbor])
    return graph


def grow_dense_high_degree_edges(graph: nx.Graph, k: int = 0, n_sel: int = None):
    assert graph is not None
    assert len(graph.nodes) > 0
    degrees = [(node, val) for (node, val) in graph.degree()]
    largest_deg = max(d for (_, d) in degrees)
    if k < 1:
        k = np.random.randint(1, max(largest_deg - 1, 2))
    assert k < len(graph.nodes) + 1
    if n_sel is None:
        n_sel = k

    for i in range(k):
        # Sort vertices by degree (v2, 3), (v4, 3), (v1, 2), (v3, 1)
        degrees = np.array(sorted([(v, d) for (v, d) in graph.degree() if d < len(graph.nodes) - 1], key=lambda x: x[1], reverse=True))
        if len(degrees) < 1:
            continue
        # Select up to n_sel highest degree vertices and choose one random one out of it
        #print(np.array(degrees)[:n_sel, 0])
        v = np.random.choice(degrees[:n_sel, 0], 1, replace=False)[0]
        exclude_neighbors = list(graph.neighbors(v))
        possible_new_neighbors = [t for t in graph.nodes if t not in exclude_neighbors]
        target = np.random.choice(possible_new_neighbors)
        graph.add_edge(v, target)

    return graph


def grow_dense_low_degree_edges(graph: nx.Graph, k: int = 0, n_sel: int = None):
    assert graph is not None
    assert len(graph.nodes) > 0
    degrees = [(node, val) for (node, val) in graph.degree()]
    smallest_deg = min(d for (_, d) in degrees)
    if k < 1:
        k = np.random.randint(1, max(smallest_deg - 1, 2))
    assert k < len(graph.nodes) + 1
    if n_sel is None:
        n_sel = k

    for i in range(k):
        # Sort vertices by degree (v2, 3), (v4, 3), (v1, 2), (v3, 1)
        degrees = np.array(sorted([(v, d) for (v, d) in graph.degree() if d < len(graph.nodes) - 1], key=lambda x: x[1], reverse=False))
        if len(degrees) < 1:
            continue
        # Select up to n_sel lowest degree vertices and choose one random one out of it
        #print(np.array(degrees)[:n_sel, 0])
        v = np.random.choice(degrees[:n_sel, 0], 1, replace=False)[0]
        exclude_neighbors = list(graph.neighbors(v))
        possible_new_neighbors = [t for t in graph.nodes if t not in exclude_neighbors]
        target = np.random.choice(possible_new_neighbors)
        graph.add_edge(v, target)

    return graph


def shrink_random_vertices(graph: nx.Graph, k: int = 0):
    assert graph is not None
    assert len(graph.nodes) > 2
    if k < 1:
        k = np.random.randint(1, len(graph.nodes) - 1)
    assert k > 0
    v_random = np.random.choice(list(graph.nodes), k, replace=False)
    graph.remove_nodes_from(list(v_random))

    return graph


def shrink_sparsify_highest_degrees(graph: nx.Graph, k: int = 0):
    assert graph is not None
    assert len(graph.nodes) > 1
    if k < 1:
        k = np.random.randint(1, len(graph.nodes))
    assert k > 0
    degrees = sorted([(v, d) for (v, d) in graph.degree()], key=lambda x: x[1], reverse=True)
    graph.remove_nodes_from(np.array(degrees)[:k, 0])

    return graph


def shrink_densify_lowest_degrees(graph: nx.Graph, k: int = 0):
    assert graph is not None
    assert len(graph.nodes) > 1
    if k < 1:
        k = np.random.randint(1, len(graph.nodes))
    assert k > 0
    degrees = sorted([(v, d) for (v, d) in graph.degree()], key=lambda x: x[1])
    graph.remove_nodes_from(np.array(degrees)[:k, 0])
    return graph


def shrink_densify_edge_contraction(graph: nx.Graph, k: int = 0):
    assert graph is not None
    assert len(graph.nodes) > 1
    if k < 1:
        k = np.random.randint(1, len(graph.nodes))
    assert k > 0

    for i in range(k):
        v = np.random.choice(list(graph.nodes))
        neighbors = list(graph.neighbors(v))
        if len(neighbors) > 0:
            t = np.random.choice(neighbors)
            graph = nx.contracted_nodes(graph, v, t, self_loops=False)

    return graph


def operation_modify_k_edges(graph: nx.Graph, mode = 'dense', k: int = 1):
    assert k > 0
    assert graph is not None and len(graph.nodes) > 0
    A = nx.adjacency_matrix(graph).todense()
    edge_value = 0 if mode == "dense" else 1  # use "dense" to add edges, other values to remove edges
    edge_indices = np.where(A == edge_value)
    if len(edge_indices[0]) < 1:  # There are no edges we can add/remove
        return
    if len(edge_indices[0]) < k:
        k = len(edge_indices)
    edge_choices = np.random.choice(len(edge_indices[0]), k, replace=False)  #np.random.randint(0, len(unconnected_indices[0]))
    for edge_choice in edge_choices:
        choice_source = edge_indices[0][edge_choice]
        choice_target = edge_indices[1][edge_choice]
        graph.add_edge(choice_source, choice_target)
    return graph


def op_1(graph: nx.Graph):
    # "linearization growth"
    v = np.random.choice(list(graph.nodes))
    neighbors = list(graph.neighbors(v))
    split1 = np.random.choice(neighbors, size=int(len(neighbors)/2))
    split2 = list(np.setdiff1d(neighbors, split1))
    v_idx_1 = max(graph.nodes) + 1
    v_idx_2 = v_idx_1 + 1
    graph.add_node(v_idx_1)
    graph.add_node(v_idx_2)
    graph.remove_node(v)
    graph.add_edge(v_idx_1, v_idx_2)
    graph.add_edges_from([(v_idx_1, target) for target in split1])
    graph.add_edges_from([(v_idx_2, target) for target in split2])


def op_2(graph: nx.Graph):
    # "triadic densification growth"
    v = np.random.choice(list(graph.nodes))
    neighbors = list(graph.neighbors(v))
    split1 = np.random.choice(neighbors, size=int(len(neighbors) / 3))
    neighbors_left = list(np.setdiff1d(neighbors, split1))
    split2 = np.random.choice(neighbors_left, size=int(len(neighbors_left) / 2))
    split3 = list(np.setdiff1d(neighbors_left, split2))

    v_idx_1 = max(graph.nodes) + 1
    v_idx_2 = v_idx_1 + 1
    v_idx_3 = v_idx_2 + 1

    graph.add_node(v_idx_1)
    graph.add_node(v_idx_2)
    graph.add_node(v_idx_3)
    graph.remove_node(v)
    graph.add_edge(v_idx_1, v_idx_2)
    graph.add_edge(v_idx_1, v_idx_3)
    graph.add_edge(v_idx_2, v_idx_3)
    graph.add_edges_from([(v_idx_1, target) for target in split1])
    graph.add_edges_from([(v_idx_2, target) for target in split2])
    graph.add_edges_from([(v_idx_3, target) for target in split3])


def op_3(graph: nx.Graph):
    # "bridge growth"
    assert graph is not None
    v_0 = np.random.choice(list(graph.nodes))
    neighbors = list(graph.neighbors(v_0))
    non_neighbors = np.setdiff1d(list(graph.nodes), neighbors)
    if len(non_neighbors) < 1:
        return
    v_1 = np.random.choice(non_neighbors)

    v_idx_new = max(graph.nodes) + 1

    graph.add_node(v_idx_new)
    graph.add_edge(v_0, v_idx_new)
    graph.add_edge(v_1, v_idx_new)
    graph.add_edge(v_1, v_1)



if __name__ == "__main__":
    graph = nx.Graph()
    print(graph.nodes)
    print(graph.edges)
    operation_modify_k_edges(graph)
    print(graph.edges)
    print()

    graph = nx.Graph()
    graph.add_node(0)
    print(graph.nodes)
    print(graph.edges)
    operation_modify_k_edges(graph)
    print(graph.edges)
    print()

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    print(graph.nodes)
    print(graph.edges)
    operation_modify_k_edges(graph)
    print(graph.edges)


"""
Abhishek's Extension

"""

"""
FUNCTION NAME : add_node
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -adds a new node to an existing graph or an empty graph

Returns :   -new graph with a node added in the input graph
"""


def add_node(input_graph: nx.Graph):
    assert input_graph is not None
    # target graph list
    new_graph = input_graph.copy()
    # new node number
    v_new = max(new_graph.nodes) + 1
    #add the node to the existing graph
    # get a random node from the input graph to connect to new node
   # random_node = choice(list(new_graph))
    new_graph.add_node(v_new)
    # connect random node and v_new node
    #new_graph.add_edge(v_new, random_node)
    # return target graph
    return new_graph


"""
FUNCTION NAME : add_edge
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -adds a new node to the given input graph
            -connect new node with edge to a random node in the given input graph

Returns :   -output of above task
"""


def add_edge(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # new node number
    v_new = max(new_graph) + 1
    # get a random node from the input graph to connect to new node
    random_node = choice(list(new_graph))
    # add new node to the graph
    new_graph.add_node(v_new)
    # connect random node and v_new node
    new_graph.add_edge(v_new, random_node)
    return new_graph


"""
FUNCTION NAME : add_node_high_deg_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest degree centrality from input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest degree centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_deg_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get degree centrality for each node
    deg_cntra = nx.degree_centrality(new_graph)
    # get the node with the highest degree centrality
    high_cntra = max(deg_cntra, key=deg_cntra.get)
    # new node number
    v_new = max(new_graph) + 1
    # add new node to the graph
    new_graph.add_node(v_new)
    # connect v_new node to the node with the highest centrality
    new_graph.add_edge(v_new, high_cntra)
    return new_graph


"""
FUNCTION NAME : remove_node_low_deg_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest degree centrality from input graph
            -remove a node with lowest degree centrality from the given input graph 

Returns :   output of above task
"""


def remove_node_low_deg_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get degree centrality for each node
    deg_cntra = nx.degree_centrality(new_graph)
    # get the node with the lowest degree centrality
    low_cntra = min(deg_cntra, key=deg_cntra.get)
    # remove node with low degree centrality from the graph
    new_graph.remove_node(low_cntra)
    return new_graph


"""
FUNCTION NAME : add_node_high_close_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest closeness centrality from the input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest closeness centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_close_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get closeness centrality for each node
    close_cntra = nx.closeness_centrality(new_graph)
    # get the node with the highest closeness centrality
    high_close_cntra = max(close_cntra, key=close_cntra.get)
    # new node number
    v_new = max(new_graph) + 1
    # add new node to the graph
    new_graph.add_node(v_new)
    # connect v_new node to the node with the highest closeness centrality
    new_graph.add_edge(v_new, high_close_cntra)
    return new_graph


"""
FUNCTION NAME : remove_node_low_close_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest closeness centrality from the input graph
            -remove a node with lowest closeness centrality from the given input graph

Returns :   output of above task
"""


def remove_node_low_close_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get closeness centrality for each node
    close_cntra = nx.closeness_centrality(new_graph)
    # get the node with the lowest closeness centrality
    low_close_cntra = min(close_cntra, key=close_cntra.get)
    # remove node with the lowest closeness centrality from the graph
    new_graph.remove_node(low_close_cntra)
    return new_graph


"""
FUNCTION NAME : add_node_high_bwtn_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest betweenness centrality from the input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest betweenness centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_bwtn_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get betweenness centrality for each node
    bwtn_cntra = nx.betweenness_centrality(new_graph)
    # get the node with the highest betweenness centrality
    high_bwtn_cntra = max(bwtn_cntra, key=bwtn_cntra.get)
    # new node number
    v_new = max(new_graph) + 1
    # add new node to the graph
    new_graph.add_node(v_new)
    # connect v_new node to the node with the highest betweenness centrality
    new_graph.add_edge(v_new, high_bwtn_cntra)
    return new_graph


"""
FUNCTION NAME : remove_node_low_bwtn_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest betweenness centrality from the input graph
            -remove a node with lowest betweenness centrality from the given input graph

Returns :   output of above task
"""


def remove_node_low_bwtn_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get betweenness centrality for each node
    bwtn_cntra = nx.betweenness_centrality(new_graph)
    # get the node with the lowest betweenness centrality
    low_bwtn_cntra = min(bwtn_cntra, key=bwtn_cntra.get)
    # remove node with low betweenness centrality from the graph
    new_graph.remove_node(low_bwtn_cntra)
    return new_graph


"""
FUNCTION NAME : add_node_high_ecc_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest eccentricity centrality from the input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest eccentricity centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_ecc_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get Eccentricity centrality for each node
    ecc_cntra = nx.eccentricity(new_graph)
    # get the node with the highest eccentricity centrality
    high_ecc_cntra = max(ecc_cntra, key=ecc_cntra.get)
    # new node number
    v_new = max(new_graph) + 1
    # add new node to the graph
    new_graph.add_node(v_new)
    # connect v_new node to the node with the highest eccentricity centrality
    new_graph.add_edge(v_new, high_ecc_cntra)
    return new_graph


"""
FUNCTION NAME : remove_node_low_ecc_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest eccentricity centrality from the input graph
            -remove a node with lowest eccentricity centrality from the given input graph

Returns :   output of above task
"""


def remove_node_low_ecc_cntra(input_graph: nx.Graph):
    assert input_graph is not None
    new_graph = input_graph.copy()
    # get Eccentricity centrality for each node
    ecc_cntra = nx.eccentricity(new_graph)
    # get the node with the lowest eccentricity centrality
    low_ecc_cntra = min(ecc_cntra, key=ecc_cntra.get)
    # remove node with the lowest eccentricity from the graph
    new_graph.remove_node(low_ecc_cntra)
    return new_graph
