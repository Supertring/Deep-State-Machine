import dgl
from dgl.data import DGLDataset
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import gmachine.DecisionOperation as do
import gemergence.operations as ops
from gemergence.compass import LocalCompass
from gemergence.util import count_parameters, Logger
import gmachine.GenerateGraph as gg

from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#list of operations
operations = [
    ops.add_node,
    ops.add_edge,
    ops.add_node_high_deg_cntra,
    ops.add_node_high_close_cntra,
    ops.add_node_high_bwtn_cntra,
    ops.add_node_high_ecc_cntra,
    ops.remove_node_low_deg_cntra,
    ops.remove_node_low_close_cntra,
    ops.remove_node_low_bwtn_cntra,
    ops.remove_node_low_ecc_cntra
]

#list of graph used
graphtypes = [
    gg.watts_strogatz_graph,
    gg.barabasi_albert_graph,
    gg.erdos_renyi_graph,
    gg.random_tree_graph,
    gg.complete_graph
]


#synthetic graph generation
class SyntheticDataset(DGLDataset):
    def __init__(self, graphtypes: list, operations: list, graph_generator_loop: int, num_reps_per_operation: int):
        self.graphtypes = graphtypes
        self.operations = operations
        self.graph_generator_loop = graph_generator_loop
        self.num_reps_per_operation = num_reps_per_operation
        super().__init__(name='synthetic')

    def process(self):
        self.graph_types = []
        self.input_graphs = []
        self.dgl_input_graphs = []
        self.operation_applied = []
        self.target_graphs = []
        self.dgl_target_graphs = []
        self.dgl_combined = []
        self.labels = []

        for gt in range(len(self.graphtypes)):
            for _ in range(self.graph_generator_loop):
                # graphtypes returns graphs in list, so we take 0th position graph eg: graphtypes[graph_function](n_samples)[list_position]
                nxg_source = self.graphtypes[gt](1)[0]
                for ops in range(len(self.operations)):
                    for rep in range(self.num_reps_per_operation):
                        try:
                            nxg_source_copy = nxg_source.copy()
                            nxg_target = self.operations[ops](nxg_source_copy)
                            nxg_target_copy = nxg_target.copy()

                            # to combine two graphs as it is, node numbers of target graph is changed                            #get the maximum node number
                            max_node_nxg_src = max(nxg_source_copy)
                            # get the number of nodes of target graph
                            no_node_nxg_tar = nx.number_of_nodes(nxg_target_copy)
                            # mapping dictonary to change the node number of target graph
                            mapping = {}
                            for i in range(no_node_nxg_tar):
                                mapping[i] = max_node_nxg_src + i + 1
                            # change the node numbers of nxg_target_copy
                            nxg_target_final = nx.relabel_nodes(nxg_target_copy, mapping)
                            # combine source and target graphs in one graph
                            nxg_combined = nx.compose(nxg_source_copy, nxg_target_final)

                        except:
                            print("something went wrong with graph generation")
                        else:
                            self.graph_types.append(gt)
                            self.input_graphs.append(nxg_source_copy)
                            self.target_graphs.append(nxg_target_copy)
                            self.operation_applied.append(ops)

                            dgl_src = dgl.from_networkx(nxg_source_copy, edge_attrs=None, edge_id_attr_name=None)
                            dgl_targ = dgl.from_networkx(nxg_target_copy, edge_attrs=None, edge_id_attr_name=None)

                            self.dgl_input_graphs.append(dgl_src)
                            self.dgl_target_graphs.append(dgl_targ)
                            self.dgl_combined.append(dgl.from_networkx(nxg_combined))
                            self.labels.append(ops)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.dgl_combined[i], self.labels[i]
        # return self.graph_types[i], self.input_graphs[i], self.dgl_input_graphs[i], self.operation_applied[i], self.target_graphs[i], self.dgl_target_graphs[i], self.labels[i]

    def __len__(self):
        return len(self.input_graphs)

#GraphConv Model
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, 32, allow_zero_in_degree=True)
        self.classify = nn.Linear(32, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, g):
        # use the in-degree of the node as the initial feature
        # h = g.in_degrees().view(-1,1).float()
        node_embed = nn.Embedding(g.number_of_nodes(), 60)
        h = node_embed.weight
        h = torch.nn.functional.normalize(h)
        h = self.conv1(g, h)
        h = torch.nn.functional.normalize(h)
        h = F.relu(h)
        h = torch.nn.functional.normalize(h)
        h = self.conv2(g, h)
        h = torch.nn.functional.normalize(h)
        h = F.relu(h)
        h = torch.nn.functional.normalize(h)
        # the output of the node feature after two layers of convolution
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        hg = torch.nn.functional.normalize(hg)
        y = self.classify(hg)
        return y

#pipeline for ml
def mlpipeline(parameters: list):
    dataset = parameters[0]
    batch_size = parameters[1]
    input_dim = parameters[2]
    hidden_dim = parameters[3]
    n_classes = parameters[4]
    n_epochs = parameters[5]
    lr = parameters[6]

    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)

    print("Creating train test samples")
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

    print("Learning Model")
    model = GCN(input_dim, hidden_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_ce = []
    for epoch in range(n_epochs):
        for batch_graph, labels in train_dataloader:
            pred = model(batch_graph)
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ce.append(loss)
        print("Running Eopch :", epoch)

    num_correct = 0
    num_tests = 0
    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    print("Test Accuracy:", (num_correct / num_tests) * 100)


#Create dataset, train model, test model
print("Creating dataset")
dataset = SyntheticDataset(graphtypes[0:2], operations[0:2], 50, 40)
print("Created dataset of size :", len(dataset))

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=16, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=16, drop_last=False)

#train model
print("Training Model")
model = GCN(60, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)
loss_ce = []
for epoch in range(20):
    for batch_graph, labels in train_dataloader:
        pred = model(batch_graph)
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ce.append(loss)
    print("Running Eopch :", epoch)

#test model
print("Testing Model")
num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print("Test Accuracy:", (num_correct/num_tests)*100)
