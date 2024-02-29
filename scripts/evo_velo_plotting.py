import os 
import sys
import argparse
import evolocity as evo
import anndata
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":

    if "../src" not in sys.path:
        sys.path.append("../src")

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset')
    parser.add_argument('--group', default="v_gene_family")
    parser.add_argument('--color') 
    parser.add_argument('--include_germline', action="store_true")
    parser.add_argument('--samplewise', action="store_true")           
    # parser.add_argument('--mode') 

    args = parser.parse_args()

    dataset = args.dataset
    group = args.group
    color = args.color
    include_germline = args.include_germline
    samplewise = args.samplewise
   
    model_names = ["ablang","sapiens","protbert","ESM"]

    results_folder = os.path.join("..","..","..","data",dataset,"results","plots","evo_velocity")

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    
    if include_germline and samplewise == False:
        inner_results_folder = os.path.join(results_folder,f"{group}_germline")
    elif include_germline and samplewise == True:
        inner_results_folder = os.path.join(results_folder,f"{group}_germline_samplewise")
    else:
        inner_results_folder = os.path.join(results_folder,group)

    if not os.path.isdir(inner_results_folder):
        os.mkdir(inner_results_folder)

    color_folder = os.path.join(inner_results_folder,color)

    if not os.path.isdir(color_folder):
        os.mkdir(color_folder)

    if group == "all_sequences":
        data_path = os.path.join("..","..","..","data",dataset,"evo_velo_adata_all_vgenes.pkl")
    elif not include_germline:
        data_path = os.path.join("..","..","..","data",dataset,f"evo_velo_{group}.pkl")
    elif include_germline:
        data_path = os.path.join("..","..","..","data",dataset,f"evo_velo_{group}_germline.pkl")

    with open(data_path ,"rb") as file:
        model_dict = pkl.load(file)

    for model in list(model_dict.keys()):
        obj = model_dict[model]
        if isinstance(obj,dict):
            for key in list(obj.keys()):
                try:
                    save_path = os.path.join(color_folder,f"evo_velo_embedding_{model}_{key}.png")
                    # sc.tl.umap(obj[key])
                    # evo.tl.velocity_embedding(obj[key])
                
                    if samplewise:
                        for sample in obj[key].obs["sample_id"].unique():
                            sample_indices = obj[key].obs["sample_id"] == sample 
                            plt.scatter(obj[key].obsm["X_umap"][sample_indices,0],obj[key].obsm["X_umap"][sample_indices,1])
                            plt.title(f"{model}_{key}_{sample} colored by {color}")
                            save_path = os.path.join(color_folder,f"evo_velo_embedding_{model}_{key}_{sample}.png")
                            plt.savefig(save_path, bbox_inches="tight")
                            plt.close()
                    else:
                        ax = evo.pl.velocity_embedding_stream(obj[key],color = color, legend_loc="right margin", show=False)
                        ax.set_title(f"{model}_{key} colored by {color}")
                        plt.savefig(save_path, bbox_inches="tight")
                except:
                    continue

        else:

            save_path = os.path.join(color_folder,f"evo_velo_embedding_{model}.png")

            sc.tl.umap(obj)
            evo.tl.velocity_embedding(obj)
            ax = evo.pl.velocity_embedding_stream(obj,color = color, legend_loc="right margin", show=False)
            ax.set_title(f"{model}_{key} colored by {color}")
            plt.savefig(save_path, bbox_inches="tight")