import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import dgl
import networkx as nx


class GraphEmbed(nn.Module):
    def __init__(self, graph_embedding_dim, graph_hidden_size):
        super(GraphEmbed, self).__init__()
        self.graph_embedding_dim = graph_embedding_dim
        self.graph_hidden_size = graph_hidden_size
        # self.embed_graph = torch.ones(1, self.graph_embedding_dim)
        self.embed_graph = nn.Embedding(1, self.graph_embedding_dim).weight
        self.conv1 = GraphConv(self.graph_embedding_dim, self.graph_hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(self.graph_hidden_size, self.graph_embedding_dim, allow_zero_in_degree=True)

    def forward(self, g):
        if g.number_of_nodes() == 0:
            return torch.zeros(1, self.graph_embedding_dim)

        else:
            # node_embed = nn.Embedding(g.number_of_nodes(), self.in_dim)
            h = self.embed_graph
            h = torch.nn.functional.normalize(h)

            h = self.conv1(g, h)
            h = torch.nn.functional.normalize(h)
            h = F.relu(h)
            h = torch.nn.functional.normalize(h)

            h = self.conv2(g, h)
            h = torch.nn.functional.normalize(h)
            h = F.relu(h)
            h = torch.nn.functional.normalize(h)
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            hg = torch.nn.functional.normalize(hg)
            self.embed_graph = torch.nn.Parameter(hg)
            return hg


# graph = nx.random_tree(30)
# dgl_graph = dgl.from_networkx(graph, edge_attrs=None, edge_id_attr_name=None)
# print(dgl_graph)
#
# ge = GraphEmbed(32, 64)
# new_embed = ge.forward(dgl_graph)
# print(new_embed)
#
# graph.add_node(31)
# graph.add_node(33)
# graph.add_node(34)
# graph.add_node(35)
# graph.add_node(36)
# graph.add_node(37)
# graph.add_node(38)
# graph.add_node(39)
# graph.add_edge(1, 31)
# graph.add_edge(2, 32)
# graph.add_edge(3, 33)
# graph.add_edge(4, 34)
# graph.add_edge(5, 35)
# graph.add_edge(7, 36)
# graph.add_edge(8, 37)
# graph.add_edge(9, 38)
# graph.add_edge(10, 39)
# dgl_graph = dgl.from_networkx(graph, edge_attrs=None, edge_id_attr_name=None)
# new_embed = ge.forward(dgl_graph)
# print(new_embed)
