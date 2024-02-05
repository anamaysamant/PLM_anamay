import umap as umap
import pandas as pd
import plotly.express as px
import sys

import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("../src")

from ablang_model import Ablang
from antiberty_model import Antiberty
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset')           # positional argument
parser.add_argument('--mode') 
parser.add_argument('--color_by')

args = parser.parse_args()

dataset = args.dataset
mode = args.mode
color_by = args.color_by

if mode == "cdr3_from_VDJ":
    suffixes = ["protbert","sapiens","ESM"]
else:
    suffixes = ["protbert-ft-200"]

IgG_subtypes = ["IGHG1","IGHG2B","IGHG2C","IGHG3"]

data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")

columns_to_save = ["barcode","contig_id","chain","v_gene","d_gene","j_gene","c_gene","raw_clonotype_id","raw_consensus_id"]

reducer = umap.UMAP(random_state=0, n_jobs=1,n_neighbors=50,min_dist=0.5)

for sample in os.listdir(data_folder_path):

    cellranger_path = os.path.join(data_folder_path, sample)
    # cellranger_path = os.path.join(cellranger_path, os.listdir(cellranger_path)[0])

    if not (os.path.isdir(os.path.join(cellranger_path,"umaps"))):
        os.mkdir(os.path.join(cellranger_path,"umaps"))

    umaps_folder = os.path.join(cellranger_path,"umaps")
    
    # repertoire_file = repertoire_file.iloc[:500,:]

    if not (os.path.isdir(os.path.join(umaps_folder,mode))):
        os.mkdir(os.path.join(umaps_folder,mode))

    save_path = os.path.join(umaps_folder,mode)
         
    save_path = os.path.abspath(save_path)
    
    for i,model in enumerate(suffixes):
        try:
            save_filepath = os.path.join(save_path,f"umap_{suffixes[i]}_{color_by}.png")

            # if os.path.exists(save_filepath):
            #     continue
            
            save_filepath = os.path.join(save_path,f"umap_{suffixes[i]}_{color_by}.png")

            embeddings_path = os.path.join(cellranger_path,"embeddings",mode,f"embeddings_{model}.csv.gzip")

            embeddings_file = pd.read_csv(embeddings_path, compression="gzip")
            embeddings_file = embeddings_file.loc[embeddings_file["chain"] == "IGH",:].reset_index(drop=True)
            embeddings_file["c_gene"] = embeddings_file["c_gene"].replace(IgG_subtypes,"IGHG")
            embeddings_file["v_gene_family"] = embeddings_file["v_gene"].apply(lambda x: x.split('-')[0])

            embedding_cols = [col for col in list(embeddings_file.columns) if col.startswith("dim")]
            metadata_cols = list(set(embeddings_file.columns) - set(embedding_cols))


            embeddings_file = embeddings_file.drop_duplicates(["c_gene"] + embedding_cols).reset_index(drop=True)
            embeddings_file = embeddings_file.dropna(subset=embedding_cols)

            
            umap_data_inv = embeddings_file[embedding_cols].copy()
            metadata = embeddings_file[metadata_cols].copy()
            del(embeddings_file)

            proj = reducer.fit_transform(umap_data_inv)
            proj = pd.DataFrame(proj)
            proj = pd.concat([metadata,proj], axis=1)

            proj_columns = list(proj.columns)

            fig_2d = px.scatter(
                proj, x=proj_columns[-2], y=proj_columns[-1], color = color_by, labels = color_by,
                color_discrete_sequence=px.colors.qualitative.Alphabet, title=f"{model}_{mode}"
            )

            fig_2d.update_layout(
                plot_bgcolor='white',
            )

            fig_2d.update_traces(marker=dict(
                                line=dict(width=0.2,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))
            
            fig_2d.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            fig_2d.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            fig_2d.write_image(save_filepath)
        except:
            continue
                        
                    


