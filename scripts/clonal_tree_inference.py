import os 
import sys
import argparse
import evolocity as evo
import anndata
import pandas as pd
import numpy as np
import torch
import scanpy as sc

if __name__ == "__main__":

    if "../src" not in sys.path:
        sys.path.append("../src")

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset')      
    # parser.add_argument('--mode') 

    args = parser.parse_args()

    dataset = "OVA_mouse"
    mode = "full_VDJ"
   
    model_names = ["ablang","sapiens","protbert","ESM"]
    data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))

    data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")

    IgG_subtypes = ["IGHG1","IGHG2B","IGHG2C","IGHG3"]

    model_dict = {}

    for i, model in enumerate(model_names):

        pooled_embeds = []
        v_fam_dict = {}
        
        for sample in pd.unique(data["sample_id"]):
            
            cellranger_path = os.path.join(data_folder_path, sample)   
            embeddings_path = os.path.join(cellranger_path,"embeddings",mode,f"embeddings_{model}.csv.gzip")

            embeddings_file = pd.read_csv(embeddings_path, compression="gzip")
            embeddings_file = embeddings_file.loc[embeddings_file["chain"] == "IGH",:].reset_index(drop=True)
            embeddings_file["c_gene"] = embeddings_file["c_gene"].replace(IgG_subtypes,"IGHG")
            embeddings_file["v_gene_family"] = embeddings_file["v_gene"].apply(lambda x: x.split('-')[0])
            embeddings_file["sample_id"] = sample

            embedding_cols = [col for col in list(embeddings_file.columns) if col.startswith("dim")]
            metadata_cols = list(set(embeddings_file.columns) - set(embedding_cols))

            embeddings_file = embeddings_file.drop_duplicates(embedding_cols).reset_index(drop=True)
            embeddings_file = embeddings_file.dropna(subset=embedding_cols)

            pooled_embeds += embeddings_file.to_dict(orient="records")

        pooled_embeds = pd.DataFrame(pooled_embeds)

        embedding_cols = [col for col in list(pooled_embeds.columns) if col.startswith("dim")]
        metadata_cols = list(set(pooled_embeds.columns) - set(embedding_cols))

        pooled_embeds["barcode"] = pooled_embeds["barcode"].apply(lambda x: x.split("-")[0])
        pooled_embeds = pooled_embeds.merge(data, on="barcode",suffixes=('', '_y'))

        only_embeddings = pooled_embeds[embedding_cols].copy()
        metadata = pooled_embeds[metadata_cols+["full_VDJ_aa"]].copy()
        
        for v_fam in pd.unique(pooled_embeds["v_gene_family"]):

                try:
                    torch.cuda.empty_cache()
                    
                    only_embeddings_v_fam = only_embeddings.loc[metadata["v_gene_family"] == v_fam, :].reset_index(drop=True)
                    metadata_v_fam = metadata.loc[metadata["v_gene_family"] == v_fam, :].reset_index(drop=True)

                    print(only_embeddings_v_fam.shape[0], metadata_v_fam.shape[0])
                                    
                    adata = anndata.AnnData(only_embeddings_v_fam)

                    adata.obs["seq"] = list(metadata_v_fam["full_VDJ_aa"])
                    adata.obs["barcode"] = list(metadata_v_fam["barcode"])
                    adata.obs["v_gene_family"] = list(metadata_v_fam["v_gene_family"])
                    adata.obs["v_gene_family_encoded"] = pd.factorize(adata.obs["v_gene_family"])[0]
                    adata.obs["sample_id"] = list(metadata_v_fam["sample_id"]) 
                    adata.obs["sample_id_encoded"] = pd.factorize(adata.obs["sample_id"])[0]
                    adata.obs["sample_clonotype"] = metadata_v_fam.apply(lambda x: x["sample_id"] + "_" + x["raw_clonotype_id"], axis = 1)
                    adata.obs["sample_clonotype_encoded"] = pd.factorize(adata.obs["sample_clonotype"])[0]
                    adata.obs["isotype"] = list(metadata_v_fam["c_gene"])
                    adata.obs["isotype"] = pd.factorize(adata.obs["isotype"])[0]
                
                    evo.pp.neighbors(adata)

                    if model == "ESM":
                        evo.tl.velocity_graph(adata)
                    else:
                        evo.tl.velocity_graph(adata, model_name=model)
        
                    del adata.uns['model']
                    
                    v_fam_dict[v_fam] = adata
                    del(adata, only_embeddings_v_fam, metadata_v_fam)
                except:
                    continue

        model_dict[model] = v_fam_dict