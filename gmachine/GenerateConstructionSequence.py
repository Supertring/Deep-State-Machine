import random
import networkx as nx
import pickle


def _random_subset(seq, m, rng):
    targets = set()
    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets



def generate_ba_sequence(n, m, seed=None):
    sequence = []
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    # Default initial graph : star graph on (m + 1) nodes
    # m edges, m+1 nodes///// or if m nodes then m-1 edges
    G = nx.star_graph(m)
    sequence.extend([0] * (m + 1))
    edges = G.edges
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        sequence.append(1)
        sequence.append(src)
        sequence.append(dst)

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # add a new node in the sequence
        sequence.append(0)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        # pair of nodes to be connected, ie. list of edges
        edge_list = list(zip([source] * m, targets))
        for edge in edge_list:
            src = edge[0]
            dst = edge[1]
            sequence.append(1)
            sequence.append(src)
            sequence.append(dst)

        source += 1
    return sequence


def ba_model_sequence_dataset(
        graph_min_size: int,
        graph_max_size: int,
        m_min: int,
        m_max: int,
        n_samples: int,
        fname
):
    samples = []
    for _ in range(n_samples):
        n = random.randint(graph_min_size, graph_max_size)
        m = random.randint(m_min, m_max)
        seq = generate_ba_sequence(n, m)
        samples.append(seq)

    with open(fname, 'wb') as f:
        pickle.dump(samples, f)
    return samples


def seq2graph(sequence, start_node_from: int = 0):
    g = nx.Graph()
    action_idx = 0
    while action_idx < len(sequence):
        action = sequence[action_idx]
        if action == 0:
            g.add_node(start_node_from + len(g.nodes))
            action_idx += 1
        elif action == 1:
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
