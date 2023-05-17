"""
    Very hacky script to show how a simple classification model can be built which predicts edit-operations given two graphs.
"""
import json
import os
import sys
import dgl
import networkx as nx
import numpy as np
import shutil
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import trange
import random

import gemergence.operations as ops
from gemergence.compass import LocalCompass
from gemergence.dataset import generate_ws_model_construction_sequence
from gemergence.util import count_parameters, Logger
import gmachine.GenerateGraph as gg

loss_include_binary_decision = True  # seems to experimenally give no benefit so far
loss_ce_factor = 2.0
num_epochs_local = 40
num_reps_per_operation = 10
num_epochs_region = 100
num_eval_reps = 50
graph_generator_loop = 1

init_size = 5
graph_emb = 75
node_emb = 7
layers_local_emb = (13, 13, 13, 13)
layers_global_emb = (13, 13, 13, 13)
layers_sim = (13, 13, 13, 13)
layers_sim_localglobal = (13, 13, 13, 13)

date_now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
base_dir = os.path.join("dev8stage1/", date_now)
os.makedirs(base_dir)
shutil.copy(os.path.basename(__file__), os.path.join(base_dir, "0-source.py"))
sys.stdout = Logger(os.path.join(base_dir, "0-logfile.txt"))
print("Basedir", base_dir)
device = torch.device('cuda:{d_idx}'.format(d_idx=torch.randint(torch.cuda.device_count(), (1,)).item()) if torch.cuda.is_available() else 'cpu')
print("Device", device)
print("Including bin_dec?", loss_include_binary_decision)

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

graphtypes = [
    gg.watts_strogatz_graph,
    gg.barabasi_albert_graph,
    gg.erdos_renyi_graph,
    gg.random_tree_graph,
    gg.complete_graph
]


model = LocalCompass(len(operations), graph_emb, device=device, layers_local_emb=layers_local_emb, layers_global_emb=layers_global_emb, layers_sim=layers_sim, layers_sim_localglobal=layers_sim_localglobal)
model.to(device)
print("count_parameters", count_parameters(model))

const_false = torch.tensor(0, dtype=torch.float32, device=device)
const_true = torch.tensor(1, dtype=torch.float32, device=device)

loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epochwise_errors_combined = []
epochwise_errors_decision = []
epochwise_errors_sim = []
epochwise_errors_nonsim = []
epochwise_errors_binsim = []
epochwise_errors_binnonsim = []

#
# def sample_graph():
#     return nx.connected_watts_strogatz_graph(np.random.randint(20, 30), np.random.randint(5, 8), np.random.choice([0.2, 0.3]))
#
#
# def sample_construction_sequence():
#     return generate_ws_model_construction_sequence(np.random.randint(20, 30), np.random.randint(5, 8), np.random.choice([0.2, 0.3]))

#create batch dataset for training
# graphtypes : 5
# operations : 10
# graph_generator_loop : 90
# num_reps_per_operation : 5
# total graphs generated : 22500
#
def syntheticDataset(graphtypes, operations, graph_generator_loop, num_reps_per_operation):
    input_graphs = []
    operation_applied = []
    target_graphs = []
    for gt in range(len(graphtypes)):
        for _ in range(graph_generator_loop):
            for ops in range(len(operations)):
                for rep in range(num_reps_per_operation):
                    try:
                        # graphtypes returns graphs in list, so we take 0th position graph eg: graphtypes[graph_function](n_samples)[list_position]
                        nxg_source = graphtypes[gt](1)[0]
                        nxg_source_copy = nxg_source.copy()
                        nxg_target = operations[ops](nxg_source_copy)
                        nxg_target_copy = nxg_target.copy()
                    except:
                        print("something went wrong with graph generation")
                    else:
                        input_graphs.append(nxg_source_copy)
                        target_graphs.append(nxg_target_copy)
                        operation_applied.append(ops)
    print("dataset generation completed")
    return input_graphs, target_graphs, operation_applied

input_graphs, target_graphs, operation_applied = syntheticDataset(graphtypes, operations, 50, 5)
print("input_graphs :",len(input_graphs))
print("target_graphs:", len(target_graphs))
print("operations   :", len(operation_applied))

input_target_operation = list(zip(input_graphs, target_graphs, operation_applied))
random.shuffle(input_target_operation)



pbar = trange(num_epochs_local, desc="Training LocalCompass", leave=True)
graph_type = []
graph_operation = []
count = 0
train_samples = 10000
for epoch in pbar:
    #prepare for batch operation
    batch_start = 0
    batch_end = 64
    total_sample = len(input_target_operation)
    n_batch = train_samples/64
    batch_num = 0

    errors_combined = {}
    errors_decision = {}
    errors_sim = {}
    errors_nonsim = {}
    errors_binsim = {}
    errors_binnonsim = {}

    # create dictonary for errors
    for i in range(len(input_target_operation)):
        errors_combined.update({i: []})
        errors_decision.update({i: []})
        errors_sim.update({i: []})
        errors_nonsim.update({i: []})
        errors_binsim.update({i: []})
        errors_binnonsim.update({i: []})

    for ib in range(int(n_batch)):
        #nx_G = nx.barabasi_albert_graph(np.random.randint(20, 30), 3)
        #graphtypes returns graphs in list, so we take 0th position graph eg: graphtypes[graph_function](n_samples)[list_position]
        #gets the input graph
        nx_G_source = input_target_operation[ib][0].copy()
        dgl_G_source = dgl.from_networkx(nx_G_source, device=device)
        h_G_source = model.embed_graph(dgl_G_source)
        hg_G_source = model.get_global_graphemb(dgl_G_source)

        # gets the target operation label
        target_ops = input_target_operation[ib][2]

        dec_target = torch.tensor([target_ops], dtype=torch.long, device=device)
        #gets the target for input
        nx_G_nearby = input_target_operation[ib][1].copy()
        dgl_G_nearby = dgl.from_networkx(nx_G_nearby, device=device)
        h_G_nearby = model.embed_graph(dgl_G_nearby)
        hg_G_nearby = model.get_global_graphemb(dgl_G_nearby)

        #d = torch.softmax(fn_decide(G1, G2, h1_init, h2_init), dim=1)
        dec_logits = model.decide_operation(h_G_source, h_G_nearby)
        error_dec = loss_ce(dec_logits.reshape(1, -1), dec_target)

        #guess_sim = model.similarity(h_G_source, h_G_nearby)
        guess_sim = model.similarity_localglobal(h_G_source, hg_G_source, h_G_nearby, hg_G_nearby)
        error_sim = loss_mse(guess_sim, model.true_similarity(True, nx_G_source, nx_G_nearby, device=device).unsqueeze(0))
        error_binsim = loss_bce(guess_sim[:, 0], const_true.unsqueeze(0))

        nx_G_foreign = input_target_operation[ib][0].copy()
        dgl_G_foreign = dgl.from_networkx(nx_G_foreign, device=device)
        h_G_forgein = model.embed_graph(dgl_G_foreign)
        hg_G_forgein = model.get_global_graphemb(dgl_G_foreign)

        #guess_nonsim = model.similarity(h_G_source, h_G_forgein)
        guess_nonsim = model.similarity_localglobal(h_G_source, hg_G_source, h_G_forgein, hg_G_forgein)
        error_nonsim = loss_mse(guess_nonsim, model.true_similarity(False, nx_G_source, nx_G_foreign, device=device).unsqueeze(0))
        error_binnonsim = loss_bce(guess_nonsim[:, 0], const_false.unsqueeze(0))
        """error_sim = const_false
        error_binsim = const_false
        error_nonsim = const_false
        error_binnonsim = const_false"""

        if loss_include_binary_decision:
            error_combined = loss_ce_factor * error_dec + error_binsim + error_binnonsim + torch.log(1 + error_nonsim + error_sim)
        else:
            error_combined = loss_ce_factor * error_dec + torch.log(1 + error_nonsim + error_sim)

        errors_combined[batch_num].append(error_combined)
        errors_decision[batch_num].append(error_dec)
        errors_sim[batch_num].append(error_sim)
        errors_nonsim[batch_num].append(error_nonsim)
        errors_binsim[batch_num].append(error_binsim)
        errors_binnonsim[batch_num].append(error_binnonsim)
                # print(output)
        #ib starts from 0 and each batch size is 64, so batch_end-1 = 63 .. (0 to 63) = 64 samples
        if(ib == batch_end-1):
            optimizer.zero_grad()
            mean_loss = torch.mean(torch.stack(errors_combined[batch_num]))
            print(mean_loss)
            mean_loss.backward(retain_graph=True)
            optimizer.step()

            pbar.set_postfix({
                "loss": mean_loss.cpu().detach().numpy(),
                "dec": torch.mean(torch.stack(errors_decision[batch_num])).cpu().detach().numpy(),
                "sim": torch.mean(torch.stack(errors_sim[batch_num])).cpu().detach().numpy(),
                "non-sim": torch.mean(torch.stack(errors_nonsim[batch_num])).cpu().detach().numpy(),
                "sim(0/1)": torch.mean(torch.stack(errors_binsim[batch_num])).cpu().detach().numpy(),
                "non-sim(0/1)": torch.mean(torch.stack(errors_binnonsim[batch_num])).cpu().detach().numpy(),
            })
            epochwise_errors_combined.append(torch.tensor(errors_combined[batch_num]).cpu().detach().numpy())
            epochwise_errors_decision.append(torch.tensor(errors_decision[batch_num]).cpu().detach().numpy())
            epochwise_errors_sim.append(torch.tensor(errors_sim[batch_num]).cpu().detach().numpy())
            epochwise_errors_nonsim.append(torch.tensor(errors_nonsim[batch_num]).cpu().detach().numpy())
            epochwise_errors_binsim.append(torch.tensor(errors_binsim[batch_num]).cpu().detach().numpy())
            epochwise_errors_binnonsim.append(torch.tensor(errors_binnonsim[batch_num]).cpu().detach().numpy())

            #if its last batch, then end of the batch will be the last sample i.e. location equal to total sample
        if(ib == n_batch-1):
            batch_start = batch_end
            batch_end = total_sample
            batch_num += 1
        else:
            batch_start = batch_end
            batch_end =+ batch_end
            batch_num += 1

torch.save(model, os.path.join(base_dir, "local_compass.pth"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


with open(os.path.join(base_dir, "fn_losses.json"), "w+") as handle:
    epochwise_data = {
        "errors_combined": epochwise_errors_combined,
        "errors_decision": epochwise_errors_decision,
        "errors_sim": epochwise_errors_sim,
        "errors_nonsim": epochwise_errors_nonsim,
        "errors_binsim": epochwise_errors_binsim,
        "errors_binnonsim": epochwise_errors_binnonsim,
    }
    json.dump(epochwise_data, handle, cls=NumpyEncoder, indent=1)


print("Evaluation")
num_correct = 0
num_tests = 0

with torch.no_grad():
    similarities_per_op = {}
    dissimilarities_per_op = {}
    op_decisions = []
    for i in range(train_samples, len(input_target_operation)):
        cur_op = input_target_operation[i][2]
        similarities_per_op[cur_op] = []
        dissimilarities_per_op[cur_op] = []
        list_decisions = []

        #nx_G = nx.barabasi_albert_graph(np.random.randint(20, 30), 3)
        nx_G_source = input_target_operation[i][0].copy()
        dgl_G_source = dgl.from_networkx(nx_G_source, device=device)
        h_G_source = model.embed_graph(dgl_G_source)
        hg_G_source = model.get_global_graphemb(dgl_G_source)

        nx_G_nearby = input_target_operation[i][1].copy()
        dgl_G_nearby = dgl.from_networkx(nx_G_nearby, device=device)
        h_G_nearby = model.embed_graph(dgl_G_nearby)
        hg_G_nearby = model.get_global_graphemb(dgl_G_nearby)

        d = torch.softmax(model.decide_operation(h_G_source, h_G_nearby), dim=1)
        list_decisions.append(d)
        #add if results are correct
        if(cur_op == torch.argmax(d).item()):
            num_correct = num_correct + 1
        #count total test
        num_tests = num_tests + 1

        #similarity = model.similarity_measure_local(h_G_source, h_G_nearby)
        similarity = model.similarity_localglobal(h_G_source, hg_G_source, h_G_nearby, hg_G_nearby)
        similarities_per_op[cur_op].append(similarity.cpu().detach().numpy())

        nx_G_foreign = input_target_operation[i][0].copy()
        dgl_G_foreign = dgl.from_networkx(nx_G_foreign, device=device)
        h_G_foreign = model.embed_graph(dgl_G_foreign)
        hg_G_foreign = model.get_global_graphemb(dgl_G_foreign)
        #dissimilarity = model.similarity_measure_local(h_G_source, h_G_foreign)
        dissimilarity = model.similarity_localglobal(h_G_source, hg_G_source, h_G_foreign, hg_G_foreign)
        dissimilarities_per_op[cur_op].append(dissimilarity.cpu().detach().numpy())


        decisions = torch.cat(list_decisions)
        op_decisions.append(decisions)
        np_decisions = decisions.cpu().detach().numpy()
        similarities_per_op[cur_op] = np.array(similarities_per_op[cur_op])
        dissimilarities_per_op[cur_op] = np.array(dissimilarities_per_op[cur_op])
        print("Decisions on op %s" % cur_op)
        print("mean", np.round(np.mean(np_decisions, axis=0), 2))
        print("std", np.round(np.std(np_decisions, axis=0), 2))
        print("mean[sim(G1~G2)]", np.round(np.mean(similarities_per_op[cur_op]), 2))
        print("std[sim(G1~G2)]", np.round(np.std(similarities_per_op[cur_op]), 2))
        print("mean[sim(G1~GF)]", np.round(np.mean(dissimilarities_per_op[cur_op]), 2))
        print("std[sim(G1~GF)]", np.round(np.std(dissimilarities_per_op[cur_op]), 2))

print('Test accuracy:', (num_correct / num_tests)*100)