import os 
import sys
import argparse
import evolocity as evo
import anndata
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import plotly.express as px

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset')
    parser.add_argument('-m','--metric', default="ARI")      

    args = parser.parse_args()

    dataset = args.dataset
    metric = args.metric
    mode = "full_VDJ"
   
    model_names = ["ablang","sapiens","protbert","ESM","protbert-ft","protbert-ft-200"]
    clustering_methods = [SpectralClustering, AgglomerativeClustering, KMeans]
    clustering_method_names = ["Spectral","Agglomerative","KMeans"]

    data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))
    data = data.drop_duplicates(["full_VDJ_aa"]).reset_index(drop=True)
    data["sample_clonotype"] = list(data.apply(lambda x: x["sample_id"] + "_" + x["raw_clonotype_id"], axis=1))

    data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")

    IgG_subtypes = ["IGHG1","IGHG2B","IGHG2C","IGHG3"]


    avg_ARI_df = []

    for j, clust_method in enumerate([SpectralClustering, KMeans, AgglomerativeClustering]):

        avg_ARI = {}

        for i, model in enumerate(model_names):
      
            
            for _,sample in (data[["original_sample_id","sample_id"]].drop_duplicates().iterrows()):

                ARI = 0

                pooled_embeds = []
                
                cellranger_path = os.path.join(data_folder_path, sample["original_sample_id"])   
                embeddings_path = os.path.join(cellranger_path,"embeddings",mode,f"embeddings_{model}.csv.gzip")
    
                embeddings_file = pd.read_csv(embeddings_path, compression="gzip")
                embeddings_file = embeddings_file.loc[embeddings_file["chain"] == "IGH",:].reset_index(drop=True)
                embeddings_file["c_gene"] = embeddings_file["c_gene"].replace(IgG_subtypes,"IGHG")
                embeddings_file["v_gene_family"] = embeddings_file["v_gene"].apply(lambda x: x.split('-')[0])
                embeddings_file["sample_id"] = sample["sample_id"]
    
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
        
                clonotype_seq_counts =  pooled_embeds["sample_clonotype"].value_counts()
                more_than_one_clonotypes = list(clonotype_seq_counts[clonotype_seq_counts > 1].index)
                more_than_one_indicator =  pooled_embeds["sample_clonotype"].apply(lambda x: x in more_than_one_clonotypes)
                pooled_embeds =  pooled_embeds.loc[more_than_one_indicator,:].reset_index(drop=True)
        
                num_labels = len(pooled_embeds["sample_clonotype"].unique())
        
                only_embeddings = pooled_embeds[embedding_cols].copy()
                metadata = pooled_embeds[metadata_cols+["full_VDJ_aa"]].copy()
        
                clustering = clust_method(n_clusters=num_labels)
                clustering.fit(only_embeddings)
        
                ARI += adjusted_rand_score(labels_pred=clustering.labels_, labels_true=pooled_embeds["sample_clonotype"])
                sample_id = sample["sample_id"]

                avg_ARI[f"{model}_{sample_id}"] = ARI # /len(pd.unique(data["sample_id"]))

        avg_ARI_df.append(avg_ARI)

    avg_ARI_df = pd.DataFrame(avg_ARI_df)
    avg_ARI_df.index = clustering_method_names

    avg_ARI_df = avg_ARI_df.transpose()
        
    fig = px.imshow(avg_ARI_df, text_auto=True, zmax = 1, title=f"Average {metric} across all samples for various clusterings")

    fig.layout.width = 1500
    fig.layout.height = 1500
    fig.write_image(os.path.join("..","..","..","data",dataset,f"clonotype_clustering_analysis_{metric}.png"))