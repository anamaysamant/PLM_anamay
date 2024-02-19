import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("../src")

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset')    
parser.add_argument('--mode') 


args = parser.parse_args()

dataset = args.dataset
mode = args.mode

suffixes = ["ablang","protbert","sapiens","ESM"]

data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")
dataset_path = os.path.join("..","..","..","data",dataset)
save_path = (os.path.join(dataset_path,"all_samples_evo_likelihoods"))


if not (os.path.isdir(save_path)):
    os.mkdir(save_path)
    
merge_over = ["barcode","contig_id","chain","v_gene","d_gene","j_gene","c_gene","raw_clonotype_id","raw_consensus_id"]

full_evo_likelihoods = []

for sample in os.listdir(data_folder_path):
        
    for i,suffix in enumerate(suffixes):

        cellranger_path = os.path.join(data_folder_path, sample)

        evo_folder = os.path.join(cellranger_path,"evo_likelihoods")
        per_plm_sample_evo_likelihoods = pd.read_csv(os.path.join(evo_folder,mode, f"evo_likelihood_{suffix}.csv"))
        if (i == 0):
            sample_evo_likelihoods = per_plm_sample_evo_likelihoods.copy()
        else:
            sample_evo_likelihoods = pd.merge(sample_evo_likelihoods, per_plm_sample_evo_likelihoods, on=merge_over)
   
        sample_evo_likelihoods = sample_evo_likelihoods.rename(columns={"evo_likelihood":f"evo_likelihood_{suffix}"})

    sample_evo_likelihoods["original_sample_id"] = sample
    sample_evo_likelihoods = sample_evo_likelihoods.to_dict(orient="records")
    full_evo_likelihoods += sample_evo_likelihoods
   
full_evo_likelihoods = pd.DataFrame(full_evo_likelihoods)
full_evo_likelihoods.to_csv(os.path.join(save_path,f"{dataset}_all_samples_all_plms_{mode}_evo_likelihoods.csv"), index=False)

    

