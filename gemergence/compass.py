import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Sequence, Tuple, Dict, List, Union, Optional
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import TAGConv


class MessageKey(object):
    repr_vertex = 'gem_hv'
    repr_edge = 'gem_he'
    repr_message = 'gem_m'
    repr_activation = 'gem_a'


def get_node_features(graph: dgl.DGLGraph):
    return graph.ndata[MessageKey.repr_vertex]


def batchify(vec):
    return vec.reshape(-1, vec.shape[-1])


class GraphEmbed(nn.Module):
    def __init__(self, node_emb_size: int, graph_emb_size: int, size_layers: Tuple[int] = (7, 7), clz_conv=TAGConv, kwargs_conv:dict=None, use_batch_norm: bool=True):
        super(GraphEmbed, self).__init__()

        kwargs_conv = kwargs_conv if kwargs_conv is not None else {}

        self._size_graph_emb = graph_emb_size
        self._size_layers = size_layers

        self.register_buffer('_embedding_unknown_graph', torch.ones(1, self._size_graph_emb))

        self.node_to_graph = nn.Linear(node_emb_size, self._size_graph_emb)
        self._node_gating = nn.Sequential(
            nn.Linear(node_emb_size, size_layers[0]),
            nn.Sigmoid()
        )
        self._clz_conv = clz_conv
        self._clz_act = torch.nn.ReLU

        self._layers = torch.nn.ModuleList()
        cur_input_size = size_layers[0]
        for ind, size in enumerate(self._size_layers):
            self._layers.append(self._clz_conv(cur_input_size, size, **kwargs_conv, bias=not use_batch_norm))
            if use_batch_norm:
                self._layers.append(torch.nn.BatchNorm1d(size))
            self._layers.append(self._clz_act())
            cur_input_size = size

        self._layer_dense = torch.nn.Linear(self._size_layers[-1], graph_emb_size)

    @property
    def size(self):
        return self._size_graph_emb

    def forward(self, g: dgl.DGLGraph, features=None) -> torch.Tensor:
        if g.number_of_nodes() == 0:
            return self._embedding_unknown_graph

        # Node features are stored as hv in ndata.
        if features is None:
            features = g.ndata[MessageKey.repr_vertex]

        h = self._node_gating(features)
        for fn_layer in self._layers:
            h = fn_layer(g, h) if isinstance(fn_layer, self._clz_conv) else fn_layer(h)

        h = self._layer_dense(h)  # one last layer without activation
        agg = h.mean(0)  # Aggregation with mean
        #agg1 = h.max(0)  # Possibility for smoothed aggregations, learned aggregations or attention
        #agg2 = h.min(0)
        #agg3 = h.std(0)

        return agg


class LocalityGuess(nn.Module):
    def __init__(self, embedding_space, hidden_space: int):
        super(LocalityGuess, self).__init__()
        self._fn1 = nn.Linear(2*embedding_space, hidden_space)
        self._fn2 = nn.Linear(hidden_space, 1)
        self._act_inter = nn.ReLU()
        self._act = nn.Sigmoid()

    def forward(self, h_g1, h_g2) -> torch.Tensor:
        h = self._act_inter(self._fn1(torch.cat([h_g1, h_g2])))
        return self._act(self._fn2(h))


class GraphSimilarity(nn.Module):
    def __init__(self, embedding_space, size_layers: Tuple[int] = (7, 7), use_layer_norm: bool = False):
        super(GraphSimilarity, self).__init__()
        self._size_layers = size_layers
        self._layers = torch.nn.ModuleList()
        self._layers.append(torch.nn.Linear(2*embedding_space, self._size_layers[0]))
        self._clz_act_inter = nn.ReLU

        cur_input_size = size_layers[0]
        for ind, size in enumerate(self._size_layers):
            self._layers.append(nn.Linear(cur_input_size, size, bias=not use_layer_norm))
            if use_layer_norm:
                self._layers.append(torch.nn.LayerNorm(size))
            self._layers.append(self._clz_act_inter())
            cur_input_size = size

        num_properties = 12  # number of scalar values used for comparison
        self._similarity = nn.Linear(size_layers[-1], num_properties)

        self._act_pos = nn.Sigmoid()
        self._act_binary = nn.Sigmoid()

    def forward(self, h_g1, h_g2) -> torch.Tensor:
        if len(h_g1.shape) == 1:
            h_g1 = h_g1.reshape(-1, h_g1.shape[0])
        if len(h_g2.shape) == 1:
            h_g2 = h_g2.reshape(-1, h_g2.shape[0])

        h = torch.hstack([h_g1, h_g2])
        for fn_layer in self._layers:
            h = fn_layer(h)

        logits = self._similarity(h)
        guess_locality = self._act_binary(logits[:,0])
        diff_values = self._act_pos(logits[:,1:])
        return torch.hstack([guess_locality.reshape(-1, 1), diff_values])

    def get_true_similarity(self, locality: bool, g1: nx.Graph, g2: nx.Graph, device=None):
        properties = []
        locality = 1 if locality else 0
        properties.append(locality)

        diff_num_neurons = abs(g1.number_of_nodes()-g2.number_of_nodes())
        diff_num_edges = abs(g1.number_of_edges()-g2.number_of_edges())
        properties.append(diff_num_neurons)
        properties.append(diff_num_edges)

        aggs = [np.mean, np.var, np.max]

        degree = lambda g: [d for n, d in g.degree()]
        degree_g1 = degree(g1)
        degree_g2 = degree(g2)
        triangles = lambda g: list(nx.triangles(g).values())
        triangles_g1 = triangles(g1)
        triangles_g2 = triangles(g2)
        clustering = lambda g: list(nx.clustering(g).values())
        clustering_g1 = clustering(g1)
        clustering_g2 = clustering(g2)
        for agg in aggs:
            diff_degree_agg = abs(agg(degree_g1)-agg(degree_g2))
            properties.append(diff_degree_agg)
            diff_triangles_agg = abs(agg(triangles_g1)-agg(triangles_g2))
            properties.append(diff_triangles_agg)
            diff_clustering_agg = abs(agg(clustering_g1)-agg(clustering_g2))
            properties.append(diff_clustering_agg)

        #edit_distance = nx.graph_edit_distance(g1, g2)
        return torch.tensor(properties, dtype=torch.float32, device=device)


class OldCompassDecision(nn.Module):
    def __init__(self, fn_embed: GraphEmbed, num_decisions):
        super(OldCompassDecision, self).__init__()
        self._fn_embed = fn_embed
        self._fn_cosine = nn.CosineSimilarity(dim=1, eps=1e-6)  # angle
        self._fn_pnorm = torch.nn.PairwiseDistance(p=2, keepdim=True)  # distance
        self._decision = nn.Linear(4, num_decisions)
        self._act = nn.ReLU()

    def read(self, hg_0, hg_1):
        """

        :param hg_0: [B, G]
        :param hg_1: [B, G]
        :return: [B, 4] batch-wise needles
        """
        d_cos = self._fn_cosine(hg_0, hg_1).reshape(-1, 1)
        d_pnorm = self._fn_pnorm(hg_0, hg_1)
        d_norm1 = torch.norm(hg_0).reshape(-1, 1)
        d_norm2 = torch.norm(hg_1).reshape(-1, 1)
        return torch.hstack([d_cos, d_pnorm, d_norm1, d_norm2])

    def navigate(self, needle):
        return self._decision(needle)

    def forward(self, g_1: dgl.DGLGraph, g_2: dgl.DGLGraph, f1=None, f2=None) -> torch.Tensor:
        h = self._fn_embed(g_1, f1)  # current graph embedding
        z = self._fn_embed(g_2, f2)  # zukunft embedding

        batchwise_h = h.reshape(-1, self._fn_embed.size)
        batchwise_z = z.reshape(-1, self._fn_embed.size)
        needle = self.read(batchwise_h, batchwise_z)

        #print(h)
        #print(z)
        #print("compass", compass)
        return self._decision(needle)


class CompassDecision(nn.Module):
    def __init__(self, num_decisions):
        super(CompassDecision, self).__init__()
        self._fn_cosine = nn.CosineSimilarity(dim=1, eps=1e-6)  # angle
        self._fn_pnorm = torch.nn.PairwiseDistance(p=2, keepdim=True)  # distance
        self._decision = nn.Linear(4, num_decisions)
        self._act = nn.ReLU()

    def read(self, hg_0, hg_1):
        """

        :param hg_0: [B, G]
        :param hg_1: [B, G]
        :return: [B, 4] batch-wise needles
        """
        d_cos = self._fn_cosine(hg_0, hg_1).reshape(-1, 1)
        d_pnorm = self._fn_pnorm(hg_0, hg_1)
        d_norm1 = torch.norm(hg_0).reshape(-1, 1)
        d_norm2 = torch.norm(hg_1).reshape(-1, 1)
        return torch.hstack([d_cos, d_pnorm, d_norm1, d_norm2])

    def navigate(self, needle):
        return self._decision(needle)

    def forward(self, h_g1: torch.Tensor, h_g2: torch.Tensor) -> torch.Tensor:
        """

        :param hg_0: [B, G] number of batches and graph embedding size
        :param hg_1: [B, G]
        """
        if len(h_g1.shape) == 1:
            h_g1 = h_g1.reshape(-1, h_g1.shape[0])
        if len(h_g2.shape) == 1:
            h_g2 = h_g2.reshape(-1, h_g2.shape[0])
        needle = self.read(h_g1, h_g2)
        return self._decision(needle)


class OldIdeaLocalCompass(nn.Module):
    """
    Idea: AutoEncoder which learns a generative model of compass-vectors from observations based on pairs of vectors in d-dimensional space
    """
    def __init__(self, navigation_space: int, hidden_space: int):
        super(OldIdeaLocalCompass, self).__init__()
        self._enc1 = nn.Linear(in_features=navigation_space, out_features=hidden_space)
        self._enc2 = nn.Linear(in_features=hidden_space, out_features=8)
        self._bn = torch.nn.BatchNorm1d(4)

        #self._dec1 = nn.Linear(in_features=navigation_space + 4, out_features=navigation_space)
        #self._dec2 = nn.Linear(in_features=navigation_space, out_features=navigation_space)

        self._fn_cosine = nn.CosineSimilarity(dim=1, eps=1e-6)  # angle
        self._fn_pnorm = torch.nn.PairwiseDistance(p=2, keepdim=True)  # distance
        self._act = nn.ReLU()

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space [B,d]
        :param log_var: log variance from the encoder's latent space [B,d]
        :return tensor [B,d]
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, h_nav1, h_nav2 = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # h_nav1 [B, navigation_space], h_nav2 [B, navigation_space]
        # x [B, 2, navigation_space] e.g. [100, 2, 10] for 10-dim navigation space vectors
        # encoding
        #vec1 = x[:, 0, :]
        #vec2 = x[:, 1, :]
        h_mulogvar = self._enc2(self._act(self._enc1(h_nav1))).reshape(-1, 2, 4)
        h_mu = h_mulogvar[:, 0, :]
        h_logvar = h_mulogvar[:, 1, :]
        estimate = self.reparameterize(h_mu, h_logvar)

        if h_nav2 is not None:
            d_cos = self._fn_cosine(h_nav1, h_nav2).reshape(-1, 1)
            d_pnorm = self._fn_pnorm(h_nav1, h_nav2)
            d_norm1 = torch.norm(h_nav1, dim=1).reshape(-1, 1)
            d_norm2 = torch.norm(h_nav2, dim=1).reshape(-1, 1)
        else:
            d_cos = estimate[:, 0].reshape(-1, 1)
            d_pnorm = estimate[:, 1].reshape(-1, 1)
            d_norm1 = estimate[:, 2].reshape(-1, 1)
            d_norm2 = estimate[:, 3].reshape(-1, 1)

        needle = torch.hstack([d_cos, d_pnorm, d_norm1, d_norm2])
        return needle, h_mu, h_logvar

        #print(h)
        #print(z)
        #print("compass", compass)
        #vector_and_compass = torch.cat([h_nav1, compass], dim=1)
        #return self._dec2(self._act(self._dec1(vector_and_compass))), h_mu, h_logvar


class CompassReconstruct(nn.Module):
    def __init__(self, navigation_space: int, hidden_size: int = 11):
        super(CompassReconstruct, self).__init__()
        self._navigation_space = navigation_space
        self._compass = OldIdeaLocalCompass(navigation_space, 9)
        self._dec1 = nn.Linear(navigation_space + 4, hidden_size)
        self._dec2 = nn.Linear(hidden_size, navigation_space)
        self._act = nn.ReLU()

    def forward(self, v1, v2=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        nav, nav_mu, nav_logvar = self._compass(v1, v2)
        vector_and_compass = torch.cat([v1, nav], dim=1)
        return self._dec2(self._act(self._dec1(vector_and_compass))), nav_mu, nav_logvar


class CompassEncoder(nn.Module):
    def __init__(self, compass: OldCompassDecision, fn_embed: GraphEmbed):
        super(CompassEncoder, self).__init__()
        self._navigation_space = compass._fn_embed._size_graph_emb
        hidden_space = max(int(self._navigation_space/2), 5)
        self._latent_space = max(int(hidden_space/2), 5)
        self._enc1 = nn.Linear(in_features=self._navigation_space, out_features=hidden_space)
        self._enc21 = nn.Linear(in_features=hidden_space, out_features=self._latent_space)
        self._enc22 = nn.Linear(in_features=hidden_space, out_features=self._latent_space)
        self._act = nn.Tanh()

        self._dec1 = nn.Linear(in_features=self._latent_space, out_features=hidden_space)
        self._dec21 = nn.Linear(in_features=hidden_space, out_features=self._navigation_space)
        self._dec22 = nn.Linear(in_features=hidden_space, out_features=self._navigation_space)

        self._log2pi = torch.log2(torch.Tensor([np.pi]))

        self._compass = compass
        self._fn_embed = fn_embed

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space [B,d]
        :param log_var: log variance from the encoder's latent space [B,d]
        :return tensor [B,d]
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):  # x=[B,G]
        h1 = F.relu(self._enc1(x))
        return self._enc21(h1), self._enc22(h1)  # mu=[B,L] / std=[B,L]

    def decode(self, x):  # x=[B,L]
        h1 = F.relu(self._dec1(x))
        return self._dec21(h1), self._dec22(h1)  # mu=[B,G] / std=[B,G]

    def forward(self, g_1: dgl.DGLGraph, g_2: dgl.DGLGraph=None, f1=None, f2=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        h = self._fn_embed(g_1, f1)  # current graph embedding

        h_mu, h_logvar = self.encode(h)
        latent = self.reparameterize(h_mu, h_logvar)
        z_est_mu, z_est_logvar = self.decode(latent)
        z_estimate = self.reparameterize(z_est_mu, z_est_logvar)

        #print(h.shape)
        #print(z_estimate.shape)
        batchwise_h = h.reshape(-1, self._fn_embed.size)
        batchwise_z_estimate = z_estimate.reshape(-1, self._fn_embed.size)
        needle = self._compass.read(batchwise_h, batchwise_z_estimate)

        # https://discuss.pytorch.org/t/multivariate-gaussian-variational-autoencoder-the-decoder-part/58235/10
        loss_rec = 0
        KLD = 0
        if g_2 is not None:
            z = self._fn_embed(g_2, f2)  # zukunft embedding
            loss_rec = self._log2pi + z_est_logvar + (z - z_est_mu) ** 2 / (2 * torch.exp(z_est_logvar))  # log prob of reconstruction
            KLD = -0.5 * torch.sum(1 + h_logvar - h_mu.pow(2) - h_logvar.exp())  # variational lower bound?

        return self._compass.navigate(needle), loss_rec, KLD


class GraphRegionEncoder(nn.Module):
    def __init__(self, fn_embed: GraphEmbed, device=None):
        super(GraphRegionEncoder, self).__init__()
        self._navigation_space = fn_embed._size_graph_emb
        hidden_space = max(int(self._navigation_space/2), 5)
        self._latent_space = max(int(hidden_space/2), 5)
        self._enc1 = nn.Linear(in_features=self._navigation_space, out_features=hidden_space)
        self._enc21 = nn.Linear(in_features=hidden_space, out_features=self._latent_space)
        self._enc22 = nn.Linear(in_features=hidden_space, out_features=self._latent_space)
        self._act = nn.Tanh()

        self._dec1 = nn.Linear(in_features=self._latent_space, out_features=hidden_space)
        self._dec21 = nn.Linear(in_features=hidden_space, out_features=self._navigation_space)
        self._dec22 = nn.Linear(in_features=hidden_space, out_features=self._navigation_space)

        self._log2pi = torch.log2(torch.tensor([np.pi], device=device))

        self._fn_embed = fn_embed

    def reparameterize(self, mu, log_var, device=None):
        """
        :param mu: mean from the encoder's latent space [B,d]
        :param log_var: log variance from the encoder's latent space [B,d]
        :return tensor [B,d]
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std, device=device) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):  # x=[B,G]
        h1 = F.relu(self._enc1(x))
        return self._enc21(h1), self._enc22(h1)  # mu=[B,L] / std=[B,L]

    def decode(self, x):  # x=[B,L]
        h1 = F.relu(self._dec1(x))
        return self._dec21(h1), self._dec22(h1)  # mu=[B,G] / std=[B,G]

    def forward(self, g_1: dgl.DGLGraph, f1=None, device=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # deprecated, just use autoencode()
        z = self._fn_embed(g_1, f1)  # current graph embedding
        return self.autoencode(z, device=device)

    def autoencode(self, h_g1: torch.tensor, device=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        h_mu, h_logvar = self.encode(h_g1)
        latent = self.reparameterize(h_mu, h_logvar, device=device)
        z_est_mu, z_est_logvar = self.decode(latent)
        z_estimate = self.reparameterize(z_est_mu, z_est_logvar, device=device)

        #batchwise_h = z.reshape(-1, self._fn_embed.size)
        #batchwise_z_estimate = z_estimate.reshape(-1, self._fn_embed.size)

        # https://discuss.pytorch.org/t/multivariate-gaussian-variational-autoencoder-the-decoder-part/58235/10
        loss_rec = self._log2pi + z_est_logvar + (h_g1 - z_est_mu) ** 2 / (2 * torch.exp(z_est_logvar))  # log prob of reconstruction
        KLD = -0.5 * torch.sum(1 + h_logvar - h_mu.pow(2) - h_logvar.exp())  # variational lower bound?

        return z_estimate, loss_rec, KLD


class GraphRegionLocalEstimator(nn.Module):
    def __init__(self, embed_size: int, size_layers: Tuple[int] = (9, 9), use_layer_norm: bool = True):
        super(GraphRegionLocalEstimator, self).__init__()
        self._navigation_space = embed_size
        hidden_space = max(int(self._navigation_space/3), 5)
        self._lin1 = nn.Linear(in_features=2*self._navigation_space, out_features=hidden_space)
        self._lin2 = nn.Linear(in_features=hidden_space, out_features=hidden_space)
        self._lin3 = nn.Linear(in_features=hidden_space, out_features=self._navigation_space)
        self._act = nn.ReLU()

        self._size_layers = size_layers
        self._layers = torch.nn.ModuleList()
        self._layers.append(torch.nn.Linear(2*self._navigation_space, self._size_layers[0]))
        self._clz_act_inter = nn.ReLU

        cur_input_size = size_layers[0]
        for ind, size in enumerate(self._size_layers):
            self._layers.append(nn.Linear(cur_input_size, size, bias=not use_layer_norm))
            if use_layer_norm:
                self._layers.append(torch.nn.LayerNorm(size))
            self._layers.append(self._clz_act_inter())
            cur_input_size = size

        self._estimator = nn.Linear(size_layers[-1], self._navigation_space)

    def forward(self, hg_1, h_target) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        hg_1 = batchify(hg_1)
        h_target = batchify(h_target)
        h = torch.hstack([hg_1, h_target])
        for fn_layer in self._layers:
            h = fn_layer(h)
        return self._estimator(h)


class LocalCompass(nn.Module):
    def __init__(self,
                 num_operations: int,
                 size_graph_emb: int = 11,
                 size_graphemb_local_rand: int = 1,
                 device=None, size_graph_emb_global: int = 13,
                 layers_local_emb: Tuple = (7, 7),
                 layers_global_emb: Tuple = (9, 9),
                 layers_sim: Tuple = (7, 7, 7),
                 layers_sim_localglobal: Tuple = (7, 7, 7)):
        super().__init__()
        self._num_ops = num_operations
        self._size_graphemb_local_rand = size_graphemb_local_rand
        self._size_emb_graph = size_graph_emb
        self._size_graph_emb_global = size_graph_emb_global
        node_emb = 4
        self._device = device

        self._init_graph_embed = torch.ones(size_graph_emb, device=device)
        self._hg_init = torch.ones(self._size_graph_emb_global, device=device)

        self._fn_cosine = nn.CosineSimilarity(dim=1, eps=1e-6)  # angle
        self._fn_pnorm = torch.nn.PairwiseDistance(p=2, keepdim=True)  # distance

        self._fn_emb = GraphEmbed(node_emb, size_graph_emb, size_layers=layers_local_emb)
        self._fn_init = nn.Linear(self._size_graphemb_local_rand + self._size_emb_graph, node_emb)

        self._fn_global_init = nn.Linear(self._size_graphemb_local_rand + self._size_graph_emb_global, node_emb)
        self._fn_global_emb = GraphEmbed(node_emb, self._size_graph_emb_global, size_layers=layers_global_emb)

        self._fn_similarity = GraphSimilarity(self._size_emb_graph, size_layers=layers_sim)
        self._fn_similarity_localglobal = GraphSimilarity(self._size_emb_graph + self._size_graph_emb_global, size_layers=layers_sim_localglobal, use_layer_norm=True)

        self._hidden_size_decision = 13
        self._fn_dec1 = nn.Linear(4 + 2 * self._size_emb_graph, self._hidden_size_decision)
        self._fn_dec2 = nn.Linear(self._hidden_size_decision, self._hidden_size_decision)
        self._fn_dec3 = nn.Linear(self._hidden_size_decision, self._num_ops)
        self._act = nn.ReLU()

    def init_graph(self, graph: dgl.DGLGraph):
        """
        :param graph:
        :return: (G_n, E_n) G_n being the number of nodes and E_g being the embedding size of a node
        """
        num_nodes = graph.number_of_nodes()
        return self._fn_init(
            torch.hstack([
                #torch.normal(0, 1, (num_nodes, self._size_init), device=self._device),
                torch.ones((num_nodes, self._size_graphemb_local_rand), device=self._device),
                self._init_graph_embed.repeat((num_nodes)).reshape(num_nodes, self._size_emb_graph)]
            )
        )

    def get_node_init_graph(self, graph: dgl.DGLGraph, fn_init, size_graph_emb, init_embed, size_rand: int):
        """
        :param graph:
        :return: (G_n, E_n) G_n being the number of nodes and E_g being the embedding size of a node
        """
        num_nodes = graph.number_of_nodes()
        input_estimate = torch.hstack([
            #torch.normal(0, 1, (num_nodes, size_rand), device=self._device),
            torch.ones((num_nodes, size_rand), device=self._device),
            init_embed.repeat((num_nodes)).reshape(num_nodes, size_graph_emb)]
        )
        return fn_init(input_estimate)

    def get_globalemb_init(self, graph):
        return self.get_node_init_graph(graph, fn_init=self._fn_global_init, size_graph_emb=self._size_graph_emb_global, init_embed=self._hg_init, size_rand=self._size_graphemb_local_rand)

    def get_global_graphemb(self, graph: dgl.DGLGraph):
        h_init = self.get_globalemb_init(graph)
        return self._fn_global_emb(graph, h_init)

    def needle(self, hg_0, hg_1):
        """
        Directional features in the embedding space

        :param hg_0: [B, G]
        :param hg_1: [B, G]
        :return: [B, 4] batch-wise needles
        """
        d_cos = self._fn_cosine(hg_0, hg_1).reshape(-1, 1)
        d_pnorm = self._fn_pnorm(hg_0, hg_1)
        d_norm1 = torch.norm(hg_0).reshape(-1, 1)
        d_norm2 = torch.norm(hg_1).reshape(-1, 1)
        return torch.hstack([d_cos, d_pnorm, d_norm1, d_norm2])

    def embed_graph(self, graph: dgl.DGLGraph):
        h_init = self.init_graph(graph)
        return self._fn_emb(graph, h_init)

    def similarity_graphs(self, g1: dgl.DGLGraph, g2: dgl.DGLGraph):
        h_g1 = self.embed_graph(g1)
        h_g2 = self.embed_graph(g2)
        return self._fn_similarity(h_g1, h_g2)

    def similarity(self, h_g1, h_g2):
        return self._fn_similarity(h_g1, h_g2)

    def similarity_localglobal(self, hl_g1, hg_g1, hl_g2, hg_g2):
        return self._fn_similarity_localglobal(torch.hstack([hl_g1, hg_g1]), torch.hstack([hl_g2, hg_g2]))

    def similarity_measure_local(self, h_g1, h_g2):
        if len(h_g1.shape) == 1:
            h_g1 = h_g1.reshape(-1, h_g1.shape[0])
        if len(h_g2.shape) == 1:
            h_g2 = h_g2.reshape(-1, h_g2.shape[0])
        sim = self._fn_similarity(h_g1, h_g2)

        if len(sim.shape) == 1:
            sim = sim.reshape(-1, 4)
        # tt[:,0]+np.log(np.sum(tt[:,1:], axis=1))
        return sim[:,0]+torch.log(torch.sum(sim[:,1:], dim=1))

    def similarity_measure_localglobal(self, hl_g1, hg_g1, hl_g2, hg_g2):
        if len(hl_g1.shape) == 1:
            hl_g1 = hl_g1.reshape(-1, hl_g1.shape[0])
        if len(hl_g2.shape) == 1:
            hl_g2 = hl_g2.reshape(-1, hl_g2.shape[0])
        sim = self._fn_similarity_localglobal(hl_g1, hg_g1, hl_g2, hg_g2)

        if len(sim.shape) == 1:
            sim = sim.reshape(-1, 4)
        # tt[:,0]+np.log(np.sum(tt[:,1:], axis=1))
        return sim[:,0]+torch.log(torch.sum(sim[:,1:], dim=1))

    def true_similarity(self, locality: bool, g1: nx.Graph, g2: nx.Graph, device=None):
        return self._fn_similarity.get_true_similarity(locality, g1, g2, device)

    def decide_operation(self, h_g1, h_g2):
        if len(h_g1.shape) == 1:
            h_g1 = h_g1.reshape(-1, h_g1.shape[0])
        if len(h_g2.shape) == 1:
            h_g2 = h_g2.reshape(-1, h_g2.shape[0])
        x = torch.hstack([self.needle(h_g1, h_g2), h_g1, h_g2])
        return self._fn_dec3(self._act(self._fn_dec2(self._act(self._fn_dec1(x)))))
