import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PredictAction:
    def __init__(self, graph_embedding_dim, n_actions: int, actions):
        super(PredictAction, self).__init__()
        self.actions = actions
        self._pred_action = nn.Linear(graph_embedding_dim, n_actions)

    def forward_train(self, hg, action_index):
        pred_action_logits = self._pred_action(hg)
        # if self.training:
        action_index = torch.tensor([action_index])
        log_loss = F.cross_entropy(pred_action_logits, action_index)
        return log_loss

    def forward_inference(self, hg):
        pred_action_logits = self._pred_action(hg)
        action_prob = F.softmax(pred_action_logits, dim=1)
        action_id = torch.argmax(action_prob)
        return action_id


class SelectNode:
    def __init__(self, graph_embedding_dim):
        super(SelectNode, self).__init__()
        self._choose_node = nn.Linear(graph_embedding_dim, 1)

    def forward_train(self, hg, target_node_index, n_node_class):
        expand_hg = hg.expand(n_node_class, -1)
        pred_choose_node_logits = self._choose_node(expand_hg)
        choices_logits = pred_choose_node_logits.view(-1, n_node_class)
        target_node_index = torch.tensor([target_node_index])
        log_loss = F.cross_entropy(choices_logits, target_node_index) / np.log(n_node_class)
        return log_loss

    def forward_inference(self, hg, n_node_class):
        expand_hg = hg.expand(n_node_class, -1)
        pred_choose_node_logits = self._choose_node(expand_hg)
        choices_logits = pred_choose_node_logits.view(-1, n_node_class)
        choices_probs = F.softmax(choices_logits, dim=1)
        return Categorical(choices_probs).sample()


# class SelectNode:
#     def __init__(self, graph_embedding_dim, n_nodes_class: int):
#         super(SelectNode, self).__init__()
#         self.n_nodes_class = n_nodes_class
#         self._choose_node = nn.Linear(graph_embedding_dim, n_nodes_class)
#
#     def forward_train(self, hg, target_node_index):
#         pred_choose_node_logits = self._choose_node(hg)
#         target_node_index = torch.tensor([target_node_index])
#         log_loss = F.cross_entropy(pred_choose_node_logits, target_node_index) / np.log(self.n_nodes_class)
#         return log_loss
#
#     def forward_inference(self, hg):
#         pred_choose_node_logits = self._choose_node(hg)
#         choose_node_prob = F.softmax(pred_choose_node_logits, dim=1)
#         predicted_node = torch.argmax(choose_node_prob)
#         return predicted_node