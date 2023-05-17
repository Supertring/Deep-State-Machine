import json
import time

import networkx as nx
import matplotlib.pyplot as plt
from grave import plot_network
from grave.style import use_attributes

import dgl
import torch.optim.sgd
from tqdm import tqdm, trange
from gmachine.deepmachines import DarmGC

from gmachine.dataset import ConstructionAction, generate_ba_model_construction_sequence, construction_sequence_to_graph


def generate_seqs_barabasi_albert_process(param_dataset_size: int, param_model_num_v: int, param_model_ba_m: int, *args,
                                          **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(generate_ba_model_construction_sequence(param_model_num_v, param_model_ba_m))
    return seqs


param_v_max = 150
param_v_min = 0
param_node_hidden_size = 35
param_n_prop_round = 2
param_learning_rate = 0.0001
param_train_epochs = 3
model_file_path = "../datasets/trained_model.pth"
param_sequences_num = 20

#param_dataset_size = 50  # number of graphs in training set
param_model_num_v = 83  # number of vertices used for model training
#param_model_ba_m = 3  # used for barabasi albert model

#dataset = generate_seqs_barabasi_albert_process(param_dataset_size, param_model_num_v, param_model_ba_m)

#
dir_path = 'datasets/sequences/'
file_path_A = dir_path + 'barabasi-albert-fixed-size-graph-construction-sequence_A.json'

same_size_construction_sequence_A = open(file_path_A)
read_construction_sequence_A = json.load(same_size_construction_sequence_A)
dataset = read_construction_sequence_A['construction_sequence'][str(param_model_num_v)]

"""Model Initialization"""
print("Initializing model...")
model = DarmGC(v_max=param_v_max, node_hidden_size=param_node_hidden_size, num_prop_rounds=param_n_prop_round)
optimizer = torch.optim.SGD(model.parameters(), lr=param_learning_rate)
model.to(device='cpu')
model.train()

"""Model Training"""
print("Training model...")
with tqdm(total=len(dataset) * param_train_epochs) as pbar:
    for epoch in range(param_train_epochs):
        for seq_idx, seq in enumerate(dataset):
            train_seq = seq
            optimizer.zero_grad()

            empty_graph = dgl.DGLGraph()
            empty_graph.to(device='cpu')

            log_loss = model.forward(empty_graph, train_seq)
            log_loss.backward()
            optimizer.step()

            # Update progress bar after each graph processing
            pbar.set_description('Graph <{g_idx}> Epoch {epoch}/{total_epochs}'.format(g_idx=seq_idx, epoch=epoch + 1,
                                                                                       total_epochs=param_train_epochs))
            pbar.set_postfix(loss=log_loss)
            pbar.update()

print("Training Completed")

print("Saving the model ...")
torch.save(model.state_dict(), model_file_path)
print("Model saved.")

"""Generating sequence"""
model.v_min = param_v_min
model.eval()
print("Started Generation")
sequences = []
for _ in range(param_sequences_num):
    empty_graph = dgl.DGLGraph()
    empty_graph.to(device='cpu')
    graph, sequence = model.forward(empty_graph)
    sequences.append(sequence)

for seq in sequences:
    print(seq)

# Save the generated sequences
generated_dataset = {
    'creation_time': time.time(),
    'model': 'generated_ba__seqs_process',
    'size': param_sequences_num,
    'construction_sequence': sequences
}
data_path = '../datasets/'
generated_file_path = data_path + 'generated_ba_seqs_process.json'
with open(generated_file_path, 'r+') as handle:
    json.dump(generated_dataset, handle, indent=1)
