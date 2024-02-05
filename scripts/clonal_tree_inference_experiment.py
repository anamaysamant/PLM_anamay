import os 
import argparse
import evolocity as evo
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # if "../src" not in sys.path:
    #     sys.path.append("../src")

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset')      
    # parser.add_argument('--mode') 

    args = parser.parse_args()

    dataset = args.dataset
    mode = "full_VDJ"
   
    model_names = ["ablang","sapiens","protbert","ESM"]

    data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))
    data = data.drop_duplicates(["full_VDJ_aa"]).reset_index(drop=True)
    data["sample_clonotype"] = list(data.apply(lambda x: x["sample_id"] + "_" + x["raw_clonotype_id"], axis=1))

    data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")

    plots_folder = os.path.join("..","..","..","data",dataset,"results","plots","alt_clonal_lineage_plots")

    if not os.path.isdir(plots_folder):
        os.mkdir(plots_folder)

    IgG_subtypes = ["IGHG1","IGHG2B","IGHG2C","IGHG3"]


    for i, model in enumerate(model_names):

        germline_embeddings = os.path.join("..","..","..","data",dataset,"all_germline_embeddings",f"all_germline_embeddings_{model}.csv.gzip")
        germline_embeddings = pd.read_csv(germline_embeddings, compression="gzip")

        germline_embeddings["sample_clonotype"] = list(germline_embeddings.apply(lambda x: x["sample_id"] + "_" + x["raw_clonotype_id"], axis=1))
        
        for _,sample in (data[["original_sample_id","sample_id"]].drop_duplicates().iterrows()):

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
            metadata = pooled_embeds[metadata_cols+["full_VDJ_aa","sample_clonotype"]].copy()
    
            clustering = KMeans(n_clusters=num_labels)
            clustering.fit(only_embeddings)

            top_k_labels = list(pd.Series(clustering.labels_).value_counts()[:10].index)

            for ind, i in enumerate(top_k_labels):

                cur_clonotype_embeds = only_embeddings.loc[clustering.labels_ == i, :].reset_index(drop=True)
                cur_clonotypes_metadata = metadata.loc[clustering.labels_ == i, :].reset_index(drop=True)

                cur_clonotype_enclone = str(cur_clonotypes_metadata["sample_clonotype"].value_counts()[0])

                cur_germline_row = germline_embeddings.loc[germline_embeddings["sample_clonotype"] == cur_clonotype_enclone, :]
                cur_germline_embedding = cur_germline_row[embedding_cols].copy()

                cur_clonotype_embeds = pd.concat((cur_germline_embedding,cur_clonotype_embeds))
                n_vertices = cur_clonotype_embeds.shape[0]

                edge_list = [(j,k) for j in range(n_vertices) for k in range(j+1,n_vertices)]

                distance_matrix = squareform(pdist(cur_clonotype_embeds.values, metric='euclidean'))

                g = ig.Graph(n_vertices, edge_list)

                g["title"] = f"{sample['sample_id']}_clonotype{ind}_{model}"
                g.vs["name"] = list(cur_clonotypes_metadata["barcode"])
                g.vs[0]["name"] = "germline"
                g.es["euclidean_distance"] = [distance_matrix[j,k] for j in range(n_vertices) for k in range(j+1,n_vertices)]

                mst_edges = g.spanning_tree(weights=g.es["euclidean_distance"], return_tree=False)

                g.es["color"] = "lightgray"
                g.es[mst_edges]["color"] = "lightblue"
                g.es["width"] = 0
                g.es[mst_edges]["width"] = 3.0

                eid_list = g.get_eids(edge_list)
                g.delete_edges([edge for edge in eid_list if edge not in mst_edges])
                fig, ax = plt.subplots(figsize=(10,10))

                ax.invert_yaxis()

                ig.plot(
                    g,
                    target=ax,
                    layout= g.layout_reingold_tilford(mode="out", root=["germline"]),
                    vertex_size=0.3,
                    vertex_color="yellow",
                    vertex_frame_width=0.1,
                    vertex_frame_color="white",
                    vertex_label=g.vs["name"],
                    vertex_label_size=5,
                    ylim = [1,-1]
                    # edge_width=[2 if married else 1 for married in g.es["married"]],
                    # edge_color=["#7142cf" if married else "#AAA" for married in g.es["married"]]
                )

                fig.savefig(os.path.join(plots_folder,f'{g["title"]}.png'))


