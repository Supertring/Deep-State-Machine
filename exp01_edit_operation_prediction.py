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

import gemergence.operations as ops
from gemergence.compass import LocalCompass
from gemergence.dataset import generate_ws_model_construction_sequence
from gemergence.util import count_parameters, Logger

loss_include_binary_decision = True  # seems to experimenally give no benefit so far
loss_ce_factor = 2.0
num_epochs_local = 2000
num_reps_per_operation = 50
num_epochs_region = 100
num_eval_reps = 100

init_size = 5
graph_emb = 17
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
    ops.shrink_sparsify_highest_degrees,
    ops.shrink_densify_lowest_degrees,
    ops.shrink_densify_edge_contraction,
    ops.grow_dense_high_degree_edges,
    ops.grow_dense_low_degree_edges
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


def sample_graph():
    return nx.connected_watts_strogatz_graph(np.random.randint(20, 30), np.random.randint(5, 8), np.random.choice([0.2, 0.3]))


def sample_construction_sequence():
    return generate_ws_model_construction_sequence(np.random.randint(20, 30), np.random.randint(5, 8), np.random.choice([0.2, 0.3]))


pbar = trange(num_epochs_local, desc="Training LocalCompass", leave=True)
for epoch in pbar:
    optimizer.zero_grad()

    #nx_G = nx.barabasi_albert_graph(np.random.randint(20, 30), 3)
    nx_G_source = sample_graph()
    dgl_G_source = dgl.from_networkx(nx_G_source, device=device)
    h_G_source = model.embed_graph(dgl_G_source)
    hg_G_source = model.get_global_graphemb(dgl_G_source)

    errors_combined = []
    errors_decision = []
    errors_sim = []
    errors_nonsim = []
    errors_binsim = []
    errors_binnonsim = []
    for rep in range(len(operations)*num_reps_per_operation):
        choice_op = np.random.randint(len(operations))
        dec_target = torch.tensor([choice_op], dtype=torch.long, device=device)

        nx_G_nearby = nx_G_source.copy()
        nx_G_nearby = operations[choice_op](nx_G_nearby)
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

        nx_G_foreign = sample_graph()
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

        errors_combined.append(error_combined)
        errors_decision.append(error_dec)
        errors_sim.append(error_sim)
        errors_nonsim.append(error_nonsim)
        errors_binsim.append(error_binsim)
        errors_binnonsim.append(error_binnonsim)
        # print(output)

    mean_loss = torch.mean(torch.stack(errors_combined))
    print(mean_loss)
    mean_loss.backward()
    optimizer.step()

    pbar.set_postfix({
        "loss": mean_loss.cpu().detach().numpy(),
        "dec": torch.mean(torch.stack(errors_decision)).cpu().detach().numpy(),
        "sim": torch.mean(torch.stack(errors_sim)).cpu().detach().numpy(),
        "non-sim": torch.mean(torch.stack(errors_nonsim)).cpu().detach().numpy(),
        "sim(0/1)": torch.mean(torch.stack(errors_binsim)).cpu().detach().numpy(),
        "non-sim(0/1)": torch.mean(torch.stack(errors_binnonsim)).cpu().detach().numpy(),
    })
    epochwise_errors_combined.append(torch.tensor(errors_combined).cpu().detach().numpy())
    epochwise_errors_decision.append(torch.tensor(errors_decision).cpu().detach().numpy())
    epochwise_errors_sim.append(torch.tensor(errors_sim).cpu().detach().numpy())
    epochwise_errors_nonsim.append(torch.tensor(errors_nonsim).cpu().detach().numpy())
    epochwise_errors_binsim.append(torch.tensor(errors_binsim).cpu().detach().numpy())
    epochwise_errors_binnonsim.append(torch.tensor(errors_binnonsim).cpu().detach().numpy())

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

with torch.no_grad():
    similarities_per_op = {}
    dissimilarities_per_op = {}
    op_decisions = []
    for cur_op in operations:
        similarities_per_op[cur_op] = []
        dissimilarities_per_op[cur_op] = []
        list_decisions = []
        for rep in range(num_eval_reps):
            #nx_G = nx.barabasi_albert_graph(np.random.randint(20, 30), 3)
            nx_G_source = sample_graph()
            dgl_G_source = dgl.from_networkx(nx_G_source, device=device)
            h_G_source = model.embed_graph(dgl_G_source)
            hg_G_source = model.get_global_graphemb(dgl_G_source)

            nx_G_nearby = nx_G_source.copy()
            nx_G_nearby = cur_op(nx_G_nearby)
            dgl_G_nearby = dgl.from_networkx(nx_G_nearby, device=device)
            h_G_nearby = model.embed_graph(dgl_G_nearby)
            hg_G_nearby = model.get_global_graphemb(dgl_G_nearby)

            d = torch.softmax(model.decide_operation(h_G_source, h_G_nearby), dim=1)
            list_decisions.append(d)

            #similarity = model.similarity_measure_local(h_G_source, h_G_nearby)
            similarity = model.similarity_localglobal(h_G_source, hg_G_source, h_G_nearby, hg_G_nearby)
            similarities_per_op[cur_op].append(similarity.cpu().detach().numpy())

            nx_G_foreign = sample_graph()
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

