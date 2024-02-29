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

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset')
parser.add_argument('--group', default="v_gene_family")
parser.add_argument('--include_germline', action='store_true') 
parser.add_argument('--no_IGD', action='store_true') 
parser.add_argument('--no_IGM', action='store_true') 
parser.add_argument('--only_IGM', action='store_true')           
# parser.add_argument('--mode') 

args = parser.parse_args()

dataset = args.dataset
group = args.group
include_germline = args.include_germline
no_IGM = args.no_IGM
only_IGM = args.only_IGM
no_IGD = args.no_IGD

mode = "full_VDJ"

model_names = ["ablang","sapiens","protbert","ESM"]
data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))

data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")
germlines_path = os.path.join("..","..","..","data",dataset,"all_germline_embeddings")

model_dict = {}

IgG_subtypes = ["IGHG1","IGHG2B","IGHG2C","IGHG3"]
IgA_subtypes = ["IGHA1","IGHA2"]

model_dict = {}

save_file_name = "evo_velo_"

if no_IGM:
    save_file_name += "no_IGM_"
elif only_IGM:
    save_file_name += "only_IGM_"
if no_IGD:
    save_file_name += "no_IGD_"

save_file_name += group

if include_germline:
    save_file_name += "_germline.pkl"
else:
    save_file_name += ".pkl"

save_path = os.path.join("..","..","..","data",dataset,save_file_name)

for i, model in enumerate(model_names):

    pooled_embeds = []
    group_dict = {}
    
    for _,sample in (data[["original_sample_id","sample_id"]].drop_duplicates().iterrows()):
        
        cellranger_path = os.path.join(data_folder_path, sample["original_sample_id"])   
        embeddings_path = os.path.join(cellranger_path,"embeddings",mode,f"embeddings_{model}.csv.gzip")

        embeddings_file = pd.read_csv(embeddings_path, compression="gzip")
        embeddings_file = embeddings_file.loc[embeddings_file["chain"] == "IGH",:].reset_index(drop=True)
        embeddings_file["c_gene"] = embeddings_file["c_gene"].replace(IgG_subtypes,"IGHG")
        embeddings_file["v_gene_family"] = embeddings_file["v_gene"].apply(lambda x: x.split('-')[0])
        embeddings_file["sample_id"] = sample["sample_id"]

        if no_IGD:
            embeddings_file = embeddings_file.loc[embeddings_file["c_gene"] != "IGHD",:]
            embeddings_file = embeddings_file.dropna(subset=["c_gene"]).reset_index(drop=True)
        if no_IGM:
            embeddings_file = embeddings_file.loc[embeddings_file["c_gene"] != "IGHM",:]
            embeddings_file = embeddings_file.dropna(subset=["c_gene"]).reset_index(drop=True)
        elif only_IGM:
            embeddings_file = embeddings_file.loc[embeddings_file["c_gene"] == "IGHM",:]
            embeddings_file = embeddings_file.dropna(subset=["c_gene"]).reset_index(drop=True)

        embedding_cols = [col for col in list(embeddings_file.columns) if col.startswith("dim")]
        metadata_cols = list(set(embeddings_file.columns) - set(embedding_cols))

        # embeddings_file = embeddings_file.drop_duplicates(embedding_cols).reset_index(drop=True)
        embeddings_file = embeddings_file.dropna(subset=embedding_cols).reset_index(drop=True)

        pooled_embeds += embeddings_file.to_dict(orient="records")

    pooled_embeds = pd.DataFrame(pooled_embeds)
    
    pooled_embeds["barcode"] = pooled_embeds["barcode"].apply(lambda x: x.split("-")[0])
    pooled_embeds = pooled_embeds.merge(data, on="barcode",suffixes=('', '_y'))
    pooled_embeds = pooled_embeds.drop_duplicates("full_VDJ_aa").reset_index(drop=True)
    pooled_embeds = pooled_embeds.dropna(subset=embedding_cols + [f"evo_likelihood_{model}"]).reset_index(drop=True) 
    
    if include_germline:
        germline_embeddings = pd.read_csv(os.path.join(germlines_path,f"all_germline_embeddings_{model}.csv.gzip"), compression="gzip")
        germline_embeddings["full_VDJ_aa"] = germline_embeddings["VDJ_ref.aa"].apply(lambda x: x.replace("-","").replace("*",""))
        # germline_embeddings["barcode"] = germline_embeddings["barcode"].apply(lambda x: x.split("_")[1])
        germline_embeddings = germline_embeddings.merge(data, on="barcode",suffixes=('', '_y'))
        germline_embeddings["barcode"] = "germline"
        germline_embeddings["c_gene"] = germline_embeddings["c_gene"].replace(IgG_subtypes,"IGHG")
        germline_embeddings["v_gene_family"] = germline_embeddings["v_gene"].apply(lambda x: x.split('-')[0])

        germline_embeddings = germline_embeddings.drop_duplicates("full_VDJ_aa").reset_index(drop=True)
        germline_embeddings = germline_embeddings.dropna(subset=embedding_cols + [f"evo_likelihood_{model}"]).reset_index(drop=True)

        germline_embeddings = germline_embeddings.to_dict(orient="records")
     
        pooled_embeds = pooled_embeds.to_dict(orient="records")
        pooled_embeds += germline_embeddings
        pooled_embeds = pd.DataFrame(pooled_embeds)

    embedding_cols = [col for col in list(pooled_embeds.columns) if col.startswith("dim")]
    metadata_cols = list(set(pooled_embeds.columns) - set(embedding_cols))
        
    only_embeddings = pooled_embeds[embedding_cols].copy()
    metadata = pooled_embeds[metadata_cols].copy()
    
    for sub_group in pd.unique(pooled_embeds[group]):

            try:
                torch.cuda.empty_cache()
                
                only_embeddings_sub_group = only_embeddings.loc[metadata[group] == sub_group, :].reset_index(drop=True)
                metadata_sub_group = metadata.loc[metadata[group] == sub_group, :].reset_index(drop=True)

                adata = anndata.AnnData(only_embeddings_sub_group)
                adata.obs["seq"] = list(metadata_sub_group["full_VDJ_aa"])
                adata.obs["barcode"] = list(metadata_sub_group["barcode"])
                
                adata.obs["v_gene_family"] = list(metadata_sub_group["v_gene_family"])     
                

                adata.obs["sample_id"] = list(metadata_sub_group["sample_id"]) 
                                   
                adata.obs["sample_clonotype"] = list(metadata_sub_group.apply(lambda x: x["sample_id"] + "_" + x["raw_clonotype_id"], axis = 1))
                # adata.obs["sample_clonotype_encoded"] = pd.factorize(adata.obs["sample_clonotype"])[0]

                # if include_germline:
                #     germline_indicator = list(metadata_sub_group["barcode"] == "germline")
                #     adata.obs.loc[germline_indicator,"v_gene_family"] = "germline"
                #     adata.obs.loc[germline_indicator,"sample_id"] = "germline"

                adata.obs["v_gene_family"] = adata.obs["v_gene_family"].astype("category")
            
                evo.pp.neighbors(adata)

                if model == "ESM":
                    evo.tl.velocity_graph(adata)
                else:
                    evo.tl.velocity_graph(adata, model_name=model)
    
                del adata.uns['model']
                
                group_dict[sub_group] = adata 
                del(adata, only_embeddings_sub_group, metadata_sub_group)
                
            except:
                continue

    model_dict[model] = group_dict

with open(save_path ,"wb") as file:
    pkl.dump(model_dict, file)