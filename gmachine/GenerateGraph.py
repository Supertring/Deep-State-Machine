import networkx as nx
import random

n_min = 50
n_max = 100

"""
FUNCTION NAME : watts_strogatz_graph
Args    :   n_samples (int) :   number of graph samples
            n_min (int)     :   minimum number of nodes in a sample
            n_max (int)     :   maximum number of nodes in a sample
            k_min (int)     :   each node is joined with minimum k nearest neighbours in a ring topology
            k_max (int)     :   each node is joined with maximum k nearest neighbours in a ring topology
            p_min (float)   :   minimum probability of rewiring each edge
            p_max (float)   :   maximum probability of rewiring each edge
            seed (int)      :   indicator of random number generation state   
            
Task    :   generates watts strogatz graph datasets

Return  :   samples (list)  : n_samples of watts strogatz graphs with given arguments
"""


def watts_strogatz_graph(n_samples, n_min=n_min, n_max=n_max, k_min=4, k_max=10, p_min=0.1, p_max=1, seed=None):
    samples = []
    for _ in range(n_samples):
        n = random.randint(n_min, n_max)
        k = random.randint(k_min, k_max)
        p = round(random.uniform(p_min, p_max), 1)
        ws = nx.watts_strogatz_graph(n, k, p, seed)
        samples.append(ws)
    return samples


"""
FUNCTION NAME : barabasi_albert_graph
Args    :   n_samples (int) :   number of graph samples
            n_min (int)     :   minimum number of nodes in a sample
            n_max (int)     :   maximum number of nodes in a sample
            m_min (int)     :   minimum number of edges to attach from a new node to existing nodes
            m_max (int)     :   maximum number of edges to attach from a new node to existing nodes
            seed (int)      :   indicator of random number generation state   

Task    :   generates barabasi albert  graph datasets

Return  :   samples (list)  : n_samples of barabasi albert graphs with given arguments
"""


def barabasi_albert_graph(n_samples, n_min=n_min, n_max=n_max, m_min=1, m_max=10, seed=None):
    samples = []
    for _ in range(n_samples):
        n = random.randint(n_min, n_max)
        m = random.randint(m_min, m_max)
        bag = nx.barabasi_albert_graph(n, m, seed)
        samples.append(bag)
    return samples


"""
FUNCTION NAME : erdos_renyi_graph
Args    :   n_samples (int) :   number of graph samples
            n_min (int)     :   minimum number of nodes in a sample
            n_max (int)     :   maximum number of nodes in a sample
            p_min (float)   :   minimum probability of edge creation
            p_max (float)   :   maximum probability of edge creation
            seed (int)      :   indicator of random number generation state
            directed (bool) :   if true, this function returns a directed graph   

Task    :   generates erdos renyi graph datasets

Return  :   samples (list)  : n_samples of erdos renyi graphs with given arguments
"""


def erdos_renyi_graph(n_samples, n_min=n_min, n_max=n_max, p_min=0.1, p_max=0.5, seed=None, directed=False):
    samples = []
    for _ in range(n_samples):
        n = random.randint(n_min, n_max)
        p = round(random.uniform(p_min, p_max), 1)
        er = nx.erdos_renyi_graph(n, p, seed)
        samples.append(er)
    return samples


"""
FUNCTION NAME : complete_graph
Args    :   n_samples (int) :   number of graph samples
            n_min (int)     :   minimum number of nodes in a sample
            n_max (int)     :   maximum number of nodes in a sample
            create_using    :   networkx graph constructor (None: default value)

Task    :   generates complete graph datasets

Return  :   samples (list)  : n_samples of complete graphs with given arguments
"""


def complete_graph(n_samples, n_min=n_min, n_max=n_max, create_using=None):
    samples = []
    for _ in range(n_samples):
        n = random.randint(n_min, n_max)
        cg = nx.complete_graph(n, create_using)
        samples.append(cg)
    return samples


"""
FUNCTION NAME : random_tree_graph
Args    :   n_samples (int) :   number of graph samples
            n_min (int)     :   minimum number of nodes in a sample
            n_max (int)     :   maximum number of nodes in a sample
            seed (int)      :   indicator of random number generation state
            create_using    :   networkx graph constructor (None: default value)

Task    :   generates random tree graph datasets

Return  :   samples (list)  : n_samples of random tree graphs with given arguments
"""


def random_tree_graph(n_samples, n_min=n_min, n_max=n_max, seed=None, create_using=None):
    samples = []
    for _ in range(n_samples):
        n = random.randint(n_min, n_max)
        rtg = nx.random_tree(n, seed)
        samples.append(rtg)
    return samples


"""
FUNCTION NAME : grid_graph
Args    :   n_samples (int) :   number of graph samples
            n_min (int)     :   minimum number of nodes in a sample
            n_max (int)     :   maximum number of nodes in a sample
            periodic (bool) :   iterable

Task    :   generates grid graph datasets

Return  :   samples (list)  : n_samples of grid graphs with given arguments
"""


def grid_graph(n_samples, n_min, n_max, periodic=False):
    samples = []
    for _ in range(n_samples):
        d1 = random.randint(n_min, n_max)
        d2 = random.randint(n_min, n_max)
        gg = nx.grid_graph(dim=[d1, d2])
        samples.append(gg)
    return samples
