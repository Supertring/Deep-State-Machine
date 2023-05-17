from functools import partial

import torch
import torch.nn as nn
import numpy as np
from dgl.nn import GraphConv
import dgl
import torch.nn.functional as F
from torch.distributions import Categorical


class GraphFeat(object):
    vertex_repr = 'hvr'
    edge_repr = 'her'
    message_repr = 'mer'
    activation_repr = 'acr'


def get_node_features(graph: dgl.DGLGraph()):
    return graph.ndata[GraphFeat.vertex_repr]


def get_node_feature(graph: dgl.DGLGraph, node, default=None):
    return graph.nodes[node].data[GraphFeat.vertex_repr] if GraphFeat.vertex_repr in graph.nodes[node].data else default


def get_nodes_features(nodes):
    return nodes.data[GraphFeat.vertex_repr]


def get_activations(graph: dgl.DGLGraph):
    return graph.ndata[GraphFeat.activation_repr]


def set_nodes_features(graph: dgl.DGLGraph, features):
    graph.ndata[GraphFeat.vertex_repr] = features


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()
        # Set the size of graph embedding
        self.graph_hidden_size = 2 * node_hidden_size
        self.register_buffer('_embedding_unknown_graph', torch.zeros(1, self.graph_hidden_size))
        # Embed graphs : Graph embedding is a weighted sum of node embeddings under a linear transformation
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 7),
            nn.Sigmoid()
        )
        # Maps node embedding to the space of graph embedding, either can be used self.node_to_graph, or self._conv
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)
        self._conv = GraphConv(7, self.graph_hidden_size, allow_zero_in_degree=True)

    def forward(self, g: dgl.DGLGraph):
        if g.number_of_nodes() == 0:
            return self._embedding_unknown_graph
        else:
            # Node features are stored as hv in ndata.
            hvs = get_node_features(g)
            return self._conv(g, self.node_gating(hvs)).sum(0, keepdim=True)


# Update node embeddings via graph propagation
class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size):
        super(GraphProp, self).__init__()
        self.num_prop_rounds = num_prop_rounds
        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size
        message_funcs = []
        node_update_funcs = []
        self.reduce_funcs = []
        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(nn.Linear(2 * node_hidden_size + node_hidden_size, self.node_activation_hidden_size))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(nn.GRUCell(self.node_activation_hidden_size, node_hidden_size))

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        return {GraphFeat.message_repr: torch.cat([edges.src[GraphFeat.vertex_repr], edges.data[GraphFeat.edge_repr]],
                                                  dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = get_nodes_features(nodes)
        m = nodes.mailbox[GraphFeat.message_repr]
        message = torch.cat([hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).mean(1)
        return {GraphFeat.activation_repr: node_activation}

    def forward(self, g):
        if g.number_of_edges() > 0:
            for t in range(self.num_prop_rounds):
                g.update_all(message_func=self.dgmg_msg, reduce_func=self.reduce_funcs[t])
                new_node_features = self.node_update_funcs[t](get_activations(g), get_node_features(g))
                set_nodes_features(g, new_node_features)


class ChooseNodeAction(nn.Module):
    def __init__(self, graph_hidden_size, node_hidden_size):
        super(ChooseNodeAction, self).__init__()
        self._log_losses = []
        self._choosed_node = nn.Linear(graph_hidden_size + 2 * node_hidden_size, 1)

    def prepare_for_training(self):
        self._log_losses = []

    def forward(self, graph: dgl.DGLGraph, graph_state_embedding, extra_node_state=None, choose_node=None):
        num_possible_nodes = graph.number_of_nodes()
        possible_node_embedding = graph.nodes[range(num_possible_nodes)].data[GraphFeat.vertex_repr]
        per_node_extra_node_state = extra_node_state.expand(num_possible_nodes, -1)
        per_node_graph_embedding_state = graph_state_embedding.expand(num_possible_nodes, -1)
        total_embedding = torch.cat(
            [per_node_graph_embedding_state, per_node_extra_node_state, possible_node_embedding], dim=1)
        node_choice_prediction = self._choosed_node(total_embedding)
        predicted_logits = node_choice_prediction.view(-1, num_possible_nodes)
        predicted_probs = F.softmax(predicted_logits, dim=1)

        if not self.training:
            return Categorical(predicted_probs).sample()
        else:
            log_loss = F.cross_entropy(predicted_logits, torch.tensor([choose_node])) / np.log(num_possible_nodes)
            self._log_losses.append(log_loss)
            return choose_node

    @property
    def get_log_losses(self):
        if len(self._log_losses) < 1:
            return torch.tensor(0)
        return torch.stack(self._log_losses).sum()


class DeepStateModule(nn.Module):
    def __init__(self, state_embedding_size, actions):
        super(DeepStateModule, self).__init__()
        self._actions = torch.tensor(actions).view(-1, 1)
        self._log_losses = []
        # self.register_buffer('_actions', torch.tensor(actions).view(-1, 1))
        self._next_state_action = nn.Linear(state_embedding_size, len(actions))

    def prepare_for_training(self):
        self._log_losses = []

    def map_action_to_index(self, action):
        try:
            return (self._actions.view(-1) == action).nonzero()[0]
        except IndexError:
            return None

    def map_index_to_action(self, action_idx):
        return self._actions[action_idx]

    def forward(self, embedding, action=None):
        next_state_logits = self._next_state_action(embedding)

        if self.training:
            next_state_logits_loss = F.cross_entropy(next_state_logits, self.map_action_to_index(action))
            self._log_losses.append(next_state_logits_loss)
            return action
        else:
            print(next_state_logits)
            next_state_probs = F.softmax(next_state_logits, dim=1)
            print("next_state_probs", next_state_probs)
            print("\n")
            # next_action_idx = Categorical(next_state_probs).sample().item()
            next_action_idx = torch.argmax(next_state_probs)
            print("next_action_idx", next_action_idx)
            return self.map_index_to_action(next_action_idx)

    @property
    def get_log_losses(self):
        if len(self._log_losses) < 1:
            return torch.tensor(0)
        return torch.stack(self._log_losses).sum()


class DarmGCActions:
    stop = 2
    add_node = 0
    add_edge = 1


class DarmGC(nn.Module):
    v_min: int = 0

    def __init__(self, v_max, node_hidden_size, num_prop_rounds):
        super(DarmGC, self).__init__()

        self._current_step = None
        self._g = dgl.DGLGraph()
        self.v_max = v_max

        # Graph embedding
        self._graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation
        self._graph_prop = GraphProp(num_prop_rounds, node_hidden_size)

        # List of states/actions
        action_values = [
            DarmGCActions.add_node,
            DarmGCActions.add_edge,
            DarmGCActions.stop
        ]
        # self.register_buffer('_actions', torch.tensor(action_values))
        # Choose node action
        self._choosing_node = ChooseNodeAction(self._graph_embed.graph_hidden_size, node_hidden_size)

        self._actions = torch.tensor(action_values)
        self._state_adding_node = DeepStateModule(self._graph_embed.graph_hidden_size, action_values)
        self._state_adding_edge = DeepStateModule(self._graph_embed.graph_hidden_size + node_hidden_size, action_values)

        # Initialize node embedding for a graph
        self._initialize_hv = nn.Linear(self._graph_embed.graph_hidden_size, node_hidden_size)
        self.register_buffer('_init_node_activation', torch.zeros(1, 2 * node_hidden_size))
        self._initialize_he = nn.Linear(2 * node_hidden_size, node_hidden_size)
        self._unknown_node = torch.zeros(node_hidden_size)
        # self.register_buffer('_unknown_node', torch.zeros(node_hidden_size))

    def _initialize_node_repr(self, node_number, cur_graph_embedding):
        g = self._g
        hv_init = self._initialize_hv(cur_graph_embedding)
        g.nodes[node_number].data[GraphFeat.vertex_repr] = hv_init
        g.nodes[node_number].data[GraphFeat.activation_repr] = self._init_node_activation
        return hv_init

    def _initialize_edge_repr(self, src, dest):
        g = self._g
        src_repr = self._g.nodes[src].data[GraphFeat.vertex_repr]
        dest_repr = self._g.nodes[dest].data[GraphFeat.vertex_repr]
        he_init = self._initialize_he(torch.cat([src_repr, dest_repr], dim=1))
        g.edges[src, dest].data[GraphFeat.edge_repr] = he_init

    def forward_train(self, actions):
        self.prepare_for_training()
        cur_graph_embedding = self._graph_embed(self._g)
        current_state = self._state_adding_node(cur_graph_embedding, self.current_action(actions))
        last_added_node_idx = -1
        while not DarmGCActions.stop == int(current_state):
            if last_added_node_idx < 0 or DarmGCActions.add_node == int(current_state):
                self._g.add_nodes(1)
                last_added_node_idx = self._g.number_of_nodes() - 1
                self._initialize_node_repr(last_added_node_idx, cur_graph_embedding)
                current_state = self._state_adding_node(cur_graph_embedding,
                                                        self.get_action_and_increment_step(actions))

            elif DarmGCActions.add_edge == int(current_state):
                last_added_node_embedding = self._g.nodes[last_added_node_idx].data[GraphFeat.vertex_repr]

                # choose source node
                # this returns the node number from the actions sequence e.g.: [V V V E 0 1 E 0 2 V E 0 3...]
                expected_node_src = self.get_action_and_increment_step(actions)
                node_choice_src = self._choosing_node(self._g, cur_graph_embedding, last_added_node_embedding,
                                                      expected_node_src)

                # Choose destination
                embedding_src_node = self._g.nodes[expected_node_src].data[GraphFeat.vertex_repr]
                expected_node_dest = self.get_action_and_increment_step(actions)
                node_choice_dest = self._choosing_node(self._g, cur_graph_embedding, embedding_src_node,
                                                       expected_node_dest)

                # Add edge to the graph
                self._g.add_edge(node_choice_src, node_choice_dest)
                self._initialize_edge_repr(node_choice_src, node_choice_dest)

                # Update Graph Propagation
                self._graph_prop(self._g)

                # Update current graph embedding state and next action/step
                cur_graph_embedding = self._graph_embed(self._g)
                current_graph_embedding_state = torch.cat([cur_graph_embedding, last_added_node_embedding], dim=1)
                current_state = self._state_adding_edge(current_graph_embedding_state,
                                                        self.get_action_and_increment_step(actions))

        return self.get_log_loss()

    def forward_inference(self):
        cur_graph_embedding = self._graph_embed(self._g)
        current_state = self._state_adding_node(cur_graph_embedding)
        last_added_node_idx = -1
        construction_sequence = []

        # if DarmGCActions.stop == current_state and self._g.number_of_nodes() == 0:
        #     current_state = DarmGCActions.add_node

        while not DarmGCActions.stop == current_state and self._g.number_of_nodes() < self.v_max + 1:
            cur_graph_embedding = self._graph_embed(self._g)

            if last_added_node_idx < 0 or DarmGCActions.add_node == current_state:
                self._g.add_nodes(1)
                last_added_node_idx = self._g.number_of_nodes() - 1
                self._initialize_node_repr(last_added_node_idx, cur_graph_embedding)
                construction_sequence.append(DarmGCActions.add_node)
                current_state = self._state_adding_node(cur_graph_embedding)

            elif DarmGCActions.add_edge == current_state:
                last_added_node_embedding = get_node_feature(self._g, last_added_node_idx, self._unknown_node)

                # Choose source and destination nodes
                node_choice_src = self._choosing_node(self._g, cur_graph_embedding, last_added_node_embedding)
                embedding_src_node = self._g.nodes[node_choice_src].data[GraphFeat.vertex_repr]
                node_choice_dest = self._choosing_node(self._g, cur_graph_embedding, embedding_src_node)

                # Add edge to the graph if not exists
                if not self._g.has_edge_between(node_choice_src, node_choice_dest):
                    self._g.add_edge(node_choice_src, node_choice_dest)
                    self._initialize_edge_repr(node_choice_src, node_choice_dest)
                    construction_sequence.append(DarmGCActions.add_edge)
                    construction_sequence.append(int(node_choice_src))
                    construction_sequence.append(int(node_choice_dest))

                # Update graph propagation
                self._graph_prop(self._g)

                # Update current graph embedding and compute next state
                cur_graph_embedding = self._graph_embed(self._g)
                new_graph_embedding = torch.cat([cur_graph_embedding, last_added_node_embedding], dim=1)
                current_state = self._state_adding_edge(new_graph_embedding)

            # if DarmGCActions.stop == current_state and self._g.number_of_nodes() <= self.v_min:
            #     current_state = DarmGCActions.add_node

        return self._g, construction_sequence

    def forward(self, graph: dgl.DGLGraph(), actions=None):
        self._g = graph
        # Initial features for nodes and edges if exists
        self._g.set_n_initializer(dgl.init.zero_initializer)
        self._g.set_e_initializer(dgl.init.zero_initializer)

        if self.training:
            return self.forward_train(actions)
        else:
            return self.forward_inference()

    # Prepare for training, initialize log loss list for each action network
    def prepare_for_training(self):
        self._current_step = 0
        self._choosing_node.prepare_for_training()
        self._state_adding_edge.prepare_for_training()
        self._state_adding_node.prepare_for_training()

    def get_log_loss(self):
        loss = 0
        loss = self._choosing_node.get_log_losses + self._state_adding_node.get_log_losses + self._state_adding_edge.get_log_losses
        return loss
        # return torch.cat(self._choosing_node.get_log_losses).sum() \
        #        + torch.cat(self._state_adding_node.get_log_losses).sum() \
        #        + torch.cat(self._state_adding_edge.get_log_losses).sum()

    def current_action(self, actions):
        return actions[self._current_step] if self._current_step < len(actions) else DarmGCActions.stop

    def get_action_and_increment_step(self, actions):
        step = self._current_step
        self._current_step += 1
        return actions[step] if step < len(actions) else DarmGCActions.stop
