import pandas as pd
import numpy as np
import os
import sys
import argparse
import torch

sys.path.append("../src")

from ablang_model import Ablang
from antiberty_model import Antiberty
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset')   

args = parser.parse_args()

dataset = args.dataset

model_names = ["ablang","protbert","sapiens","ESM"]
model_classes = [Ablang,ProtBert, Sapiens,ESM]

abforest_files_location = os.path.join("..","..","..","data",dataset,"clonal_analysis")
save_path = os.path.join("..","..","..","data",dataset,"clonal_sub_rank_table.csv")

output_table = []

file_names = os.listdir(abforest_files_location)

node_features_filename = [file_name for file_name in file_names if ("node_features" in file_name)]
adj_mat_filename = [file_name for file_name in file_names if ("adj_mat" in file_name)]

node_features_filename = node_features_filename[0]
adj_mat_filename = adj_mat_filename[0]

for j,model in enumerate(model_names):
    
    torch.cuda.empty_cache()
    
    all_node_features = pd.read_csv(os.path.join(abforest_files_location, node_features_filename))
    all_adj_mat = pd.read_csv(os.path.join(abforest_files_location, adj_mat_filename))
    
    clonotypes = list(all_node_features["sample_clonotype"].unique())
    
    if model == "ablang":
        model_init = Ablang(chain="heavy")
    if model == "sapiens":
        model_init = Sapiens(chain_type="H")
    else:
        model_init = model_classes[j]()

    for clonotype in clonotypes:
                
        node_features = all_node_features.loc[all_node_features["sample_clonotype"] == clonotype,:]
        adj_mat = all_adj_mat.loc[all_adj_mat["sample_clonotype"] == clonotype,:]

        germline_index = node_features["label"].astype(int).max()
        
        node_features = node_features.loc[node_features["label"] != germline_index, :].reset_index(drop = True)
        adj_mat = adj_mat.loc[adj_mat["i"] != germline_index,:].reset_index(drop = True)
        
        num_edges = adj_mat.shape[0]

        mean_substitution_ranks_per_edge = []

        for i in range(adj_mat.shape[0]):
            
            seq_1 = node_features["network_sequences"][adj_mat.iloc[i,0] - 1]
            seq_2 = node_features["network_sequences"][adj_mat.iloc[i,1] - 1]

            if len(seq_1) != len(seq_2):
                continue

            diff_positions = [k for k in range(len(seq_1)) if seq_1[k] != seq_2[k]]
            prob_matrix = model_init.calc_probability_matrix(seq_1)
            
            substitute_ranks = []
            
            for pos in diff_positions:

                likelihood_values = pd.Series(prob_matrix.iloc[pos,:])
                ranks = likelihood_values.rank(ascending=False)

                substitute_ranks.append(ranks[seq_2[pos]])

            mean_substitute_rank = np.average(substitute_ranks)
            mean_substitution_ranks_per_edge.append(mean_substitute_rank)

            output_table.append({"model":model,"clonotype":clonotype,"edge":f"edge_{i}","n_subs":len(diff_positions), "mean_sub_rank":mean_substitute_rank})
    

    del(model_init)    
            
output_table = pd.DataFrame(output_table)   

output_table.to_csv(save_path, index=False)