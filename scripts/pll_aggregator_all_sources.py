import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("../src")

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset')    

args = parser.parse_args()

dataset = args.dataset

data_folder_path = os.path.join("..","..","..","data",dataset,"all_samples_evo_likelihoods")
dataset_path = os.path.join("..","..","..","data",dataset)
save_path = os.path.join(data_folder_path,f"{dataset}_all_samples_all_plms_all_sources_evo_likelihoods.csv")


full_evo_likelihoods = []

for i,file in enumerate(os.listdir(data_folder_path)):
        
    file_path = os.path.join(data_folder_path, file)
    per_source_evo_likelihoods = pd.read_csv(file_path)
    
    if (i == 0):
        all_sources_evo_likelihoods = per_source_evo_likelihoods.copy()
    else:
        all_sources_evo_likelihoods = pd.merge(all_sources_evo_likelihoods, per_source_evo_likelihoods)

all_sources_evo_likelihoods.to_csv(save_path, index=False)



