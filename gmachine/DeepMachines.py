import networkx as nx
from gmachine.Classifiers import SelectNode, PredictAction
from gmachine.GraphEmbedding import GraphEmbed
import dgl
import torch
import torch.nn as nn


# Deep Auto-Regressive Machines for Graph Construction (DARM_GC)
class Actions:
    add_node = 0
    add_edge = 1
    stop = 2


def get_current_action(actions, action_idx):
    return actions[action_idx] if action_idx < len(actions) else Actions.stop


class DARM_GC(nn.Module):
    def __init__(self, max_node):
        super(DARM_GC, self).__init__()
        self.max_node = max_node
        # Graph embedding module
        self.log_losses = None
        self.n_nodes_class = None
        self.node_classes = None
        self.current_action = None
        self.node_id = None
        self.action_index = None
        self.graph_embedding = None
        self.nx_dgl_graph = None
        self.nx_graph = None
        self.graph_embedding_dim = 16
        self.hidden_dim = 30

        # number of actions
        self.n_actions = 3
        # Action states
        self.action_values = [
            Actions.add_node,
            Actions.add_edge,
            Actions.stop
        ]
        self._graph_embed = GraphEmbed(self.graph_embedding_dim, self.hidden_dim)
        self._model_current_action = Pred
        ictAction(self.graph_embedding_dim, self.n_actions, self.action_values)
        self._model_choose_node = SelectNode(self.graph_embedding_dim)

    # Initial setup for training
    def prepare_for_initial_setup(self):
        self.nx_graph = nx.Graph()
        self.nx_dgl_graph = dgl.from_networkx(self.nx_graph, edge_attrs=None, edge_id_attr_name=None)
        self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
        self.action_index = 0
        self.node_id = 0
        self.current_action = 0
        self.node_classes = []
        self.n_nodes_class = 0
        # Initialize log loss
        self.log_losses = []

    def get_log_loss(self):
        if len(self.log_losses) < 1:
            return torch.tensor(0)
        return torch.stack(self.log_losses).sum() / (len(self.log_losses))

    def forward_train(self, actions):
        self.prepare_for_initial_setup()
        # get the actual current action from the sequence
        self.current_action = get_current_action(actions, self.action_index)
        # Initialize Models
        in_dim = len(self.graph_embedding[0])
        # Models
        # model_current_action = PredictAction(in_dim, self.n_actions, self.action_values)

        # Train till the end of the state
        while not Actions.stop == int(self.current_action):
            # create graph embedding
            self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
            # Check which actions to perform
            if Actions.add_node == int(self.current_action):
                # predict the current action and get log loss
                # get index of target action from action values
                target_action_index = self.action_values.index(Actions.add_node)
                add_node_log_loss = self._model_current_action.forward_train(self.graph_embedding, target_action_index)
                self.log_losses.append(add_node_log_loss)
                # Apply the operation on the graph
                self.nx_graph.add_node(self.node_id)
                # update basic parameters
                self.node_classes.append(self.node_id)
                self.n_nodes_class = len(self.node_classes)

                # increment node id number
                self.node_id += 1

                # Re-calculate the embedding based on the modified graph
                self.nx_dgl_graph = dgl.from_networkx(self.nx_graph, edge_attrs=None, edge_id_attr_name=None)
                self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
                # update with what next action is from the action sequence
                self.action_index += 1
                self.current_action = get_current_action(actions, self.action_index)

                # if the action is to add edge
            elif Actions.add_edge == int(self.current_action):

                # choose source node
                self.action_index += 1
                actual_src_node = get_current_action(actions, self.action_index)
                # get index of target node from the node list (ie. to define the list of classes in classifier)
                src_node_index = self.node_classes.index(actual_src_node)
                # predict source node

                # out_dim = self.n_nodes_class
                # model_choose_node = SelectNode(in_dim, self.n_nodes_class)
                add_src_loss = self._model_choose_node.forward_train(self.graph_embedding, src_node_index,
                                                                     self.nx_graph.number_of_nodes())
                self.log_losses.append(add_src_loss)
                # choose destination node
                self.action_index += 1
                actual_des_node = get_current_action(actions, self.action_index)
                # get index of target node from the node list (ie. to define the list of classes in classifier)
                des_node_index = self.node_classes.index(actual_des_node)
                # predict dest node
                add_des_loss = self._model_choose_node.forward_train(self.graph_embedding, des_node_index,
                                                                     self.nx_graph.number_of_nodes())
                self.log_losses.append(add_des_loss)

                # Add edge to the graph
                self.nx_graph.add_edge(actual_src_node, actual_des_node)
                # Re-calculate the embedding based on the modified graph
                self.nx_dgl_graph = dgl.from_networkx(self.nx_graph, edge_attrs=None, edge_id_attr_name=None)
                # generate graph embedding
                self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
                # update with what next action is from the action sequence
                self.action_index += 1
                self.current_action = get_current_action(actions, self.action_index)
        return self.get_log_loss()
        # get current action

    def forward_inference(self):
        self.prepare_for_initial_setup()
        construction_sequence = []
        # Initialize Models
        in_dim = len(self.graph_embedding[0])
        # Models
        # model_current_action = PredictAction(in_dim, self.n_actions, self.action_values)
        # self.current_action = 0 #self._model_current_action.forward_inference(self.graph_embedding)
        print("first action : ", self.current_action)

        # Train till the end of the state
        while not Actions.stop == int(self.current_action) and self.nx_graph.number_of_nodes() < self.max_node:
            print("inside while")
            # create graph embedding
            self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
            # Check which actions to perform
            if Actions.add_node == int(self.current_action):
                print("Inside Add node", self.current_action)
                self.nx_graph.add_node(self.node_id)
                # update basic parameters
                self.node_classes.append(self.node_id)
                self.n_nodes_class = len(self.node_classes)
                # increment node id number
                self.node_id += 1
                construction_sequence.append(Actions.add_node)
                # Re-calculate the embedding based on the modified graph
                self.nx_dgl_graph = dgl.from_networkx(self.nx_graph, edge_attrs=None, edge_id_attr_name=None)
                self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
                print(self.graph_embedding)
                # predict next action
                self.current_action = self._model_current_action.forward_inference(self.graph_embedding)
                print("current action", self.current_action)

                # if the action is to add edge
            elif Actions.add_edge == int(self.current_action) and self.nx_graph.number_of_nodes() >= 2:
                print("Inside add edge")
                # out_dim = self.n_nodes_class
                # model_choose_node = SelectNode(in_dim, self.n_nodes_class)
                predicted_src_node = self._model_choose_node.forward_inference(self.graph_embedding,
                                                                               self.nx_graph.number_of_nodes())
                predicted_des_node = self._model_choose_node.forward_inference(self.graph_embedding,
                                                                               self.nx_graph.number_of_nodes())

                # Add edge to graph if not exist
                if not self.nx_graph.has_edge(predicted_src_node, predicted_des_node):
                    self.nx_graph.add_edge(predicted_src_node, predicted_des_node)
                    construction_sequence.append(Actions.add_edge)
                    construction_sequence.append(int(predicted_src_node))
                    construction_sequence.append(int(predicted_des_node))

                self.nx_dgl_graph = dgl.from_networkx(self.nx_graph, edge_attrs=None, edge_id_attr_name=None)
                # generate graph embedding
                self.graph_embedding = self._graph_embed(self.nx_dgl_graph)
                self.current_action = self._model_current_action.forward_inference(self.graph_embedding)

            if Actions.stop == int(self.current_action) and self.nx_graph.number_of_nodes() < self.max_node:
                print("if stop and node less than max node")
                self.current_action = Actions.add_node
                construction_sequence.append(Actions.add_node)

            if Actions.add_edge == int(self.current_action) and self.nx_graph.number_of_nodes() < 2:
                print("if add edge and node less than 2")
                self.current_action = Actions.add_node
                construction_sequence.append(Actions.add_node)

        return self.nx_graph, construction_sequence
