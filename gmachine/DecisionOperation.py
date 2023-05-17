import networkx as nx
from random import choice
from random import choices

"""
FUNCTION NAME : add_node
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -adds a new node to an existing graph or an empty graph

Returns :   -new graph with a node added in the input graph
"""


def add_node(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input_graph
    for i in range(len(input_graph)):
        # copy i th graph
        new_graph = input_graph[i].copy()
        # new node number
        v_new = max(new_graph.nodes) + 1
        # add new node to the existing graph
        new_graph.add_node(v_new)
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph



def add_multiple_node(input_graph: nx.Graph, n=5, m=2):
    assert input_graph is not None
    # target graph list
    new_graph = input_graph.copy()
    # new nodes number
    v_max = max(new_graph.nodes) + 1
    #list of new n nodes number
    vs_new = [v for v in range(v_max, v_max+n)]
    deg_cntra = nx.degree_centrality(new_graph)
    #list of new n nodes with max degree
    max_deg_nodes = sorted(deg_cntra, key=deg_cntra.get, reverse=True)[:n-m]    
    #add the node to the existing graph
    # get a random node from the input graph to connect to new node
    # random_node = choice(list(new_graph))
    for i in range(n):
        new_graph.add_node(vs_new[i])
    #get first n-m nodes from the new nodes
    vs_new_direct = vs_new[:n-m]
    # connect vs_new node with nodes having high degree centrality in the graph
    for j in range(n-m):
        new_graph.add_edge(vs_new_direct[j],max_deg_nodes[j])
    
    vs_new_indirect = vs_new[n-m:]
    for k in range(len(vs_new_indirect)):
        new_graph.add_edge(vs_new_indirect[k], vs_new[k])
        
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
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
        # new node number
        v_new = max(new_graph) + 1
        # get a random node from the input graph to connect to new node
        random_node = choice(list(new_graph))
        # add new node to the graph
        new_graph.add_node(v_new)
        # connect random node and v_new node
        new_graph.add_edge(v_new, random_node)
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


def add_multiple_edge(input_graph: nx.Graph, n: int):
    assert input_graph is not None
    new_graph = input_graph.copy()
    deg_cntra = nx.degree_centrality(new_graph)
    #list of new n nodes with max degree
    max_deg_nodes = sorted(deg_cntra, key=deg_cntra.get, reverse=True)[:n]
    excludes_max_nodes = list(set(new_graph)-set(max_deg_nodes))
    # get a random node from the input graph to connect to new node
    random_nodes = choices(excludes_max_nodes, k=n)
    # add edge between the max deg nodes and random nodes
    for i in range(n):
        new_graph.add_edge(max_deg_nodes[i],random_nodes[i])
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
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
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
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : remove_node_low_deg_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest degree centrality from input graph
            -remove a node with lowest degree centrality from the given input graph 

Returns :   output of above task
"""


def remove_node_low_deg_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
        # get degree centrality for each node
        deg_cntra = nx.degree_centrality(new_graph)
        # get the node with the lowest degree centrality
        low_cntra = min(deg_cntra, key=deg_cntra.get)
        # remove node with low degree centrality from the graph
        new_graph.remove_node(low_cntra)
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : add_node_high_close_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest closeness centrality from the input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest closeness centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_close_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
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
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : remove_node_low_close_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest closeness centrality from the input graph
            -remove a node with lowest closeness centrality from the given input graph

Returns :   output of above task
"""


def remove_node_low_close_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
        # get closeness centrality for each node
        close_cntra = nx.closeness_centrality(new_graph)
        # get the node with the lowest closeness centrality
        low_close_cntra = min(close_cntra, key=close_cntra.get)
        # remove node with the lowest closeness centrality from the graph
        new_graph.remove_node(low_close_cntra)
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : add_node_high_bwtn_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest betweenness centrality from the input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest betweenness centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_bwtn_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
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
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : remove_node_low_bwtn_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest betweenness centrality from the input graph
            -remove a node with lowest betweenness centrality from the given input graph

Returns :   output of above task
"""


def remove_node_low_bwtn_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
        # get betweenness centrality for each node
        bwtn_cntra = nx.betweenness_centrality(new_graph)
        # get the node with the lowest betweenness centrality
        low_bwtn_cntra = min(bwtn_cntra, key=bwtn_cntra.get)
        # remove node with low betweenness centrality from the graph
        new_graph.remove_node(low_bwtn_cntra)
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : add_node_high_ecc_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with highest eccentricity centrality from the input graph
            -adds a new node in the given input graph
            -adds edge between new node and node with highest eccentricity centrality in the given input graph

Returns :   output of above task
"""


def add_node_high_ecc_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
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
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph


"""
FUNCTION NAME : remove_node_low_ecc_cntra
Args    :   input_graph (nx.Graph)  : a graph or a list of graph

Task    :   -get a node with lowest eccentricity centrality from the input graph
            -remove a node with lowest eccentricity centrality from the given input graph

Returns :   output of above task
"""


def remove_node_low_ecc_cntra(input_graph: nx.Graph):
    # target graph list
    target_graph = []
    # loop into the input graph
    for i in range(len(input_graph)):
        new_graph = input_graph[i].copy()
        # get Eccentricity centrality for each node
        ecc_cntra = nx.eccentricity(new_graph)
        # get the node with the lowest eccentricity centrality
        low_ecc_cntra = min(ecc_cntra, key=ecc_cntra.get)
        # remove node with the lowest eccentricity from the graph
        new_graph.remove_node(low_ecc_cntra)
        # append the new graph to the target graph list
        target_graph.append(new_graph)
        # return target graph
    return target_graph



