#script to get data from PlatypusDB
#Inference_script
from transformers import AutoModelForSequenceClassification
from transformers import RobertaForMaskedLM
from transformers import RobertaModel
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
import seaborn as sns
from datasets import load_dataset
import os
import torch
from inference_functions import *

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Initialise the tokeniser
tokenizer = RobertaTokenizer.from_pretrained(
    "antibody-tokenizer"
)

#read in bm data from the different folders 
#folder_names = ["TNFR2.BM.3m.S1", "TNFR2.BM.3m.S2", "TNFR2.BM.3m.S3", "TNFR2.BM.3m.S4", "TNFR2.BM.3m.S5", "TNFR2.BM.12m.S6", "TNFR2.BM.12m.S7", "TNFR2.BM.12m.S8", "TNFR2.BM.18m.S9", "TNFR2.BM.18m.S10", "TNFR2.BM.18m.S11"]
folder_names = ["TNFR2.BM.3m.S1"]
df_clones_bm = pd.DataFrame()
for name in folder_names:
    df_clones1 = pd.read_csv(f'data/agrafiotis2021a__VDJ_RAW/{name}/consensus_annotations.csv', encoding='utf-8')
    df_clones_bm = pd.concat([df_clones_bm, df_clones1])


#read in spleen data from the different folders
#folder_names = ["TNFR2.SP.3m.S12", "TNFR2.SP.3m.S13", "TNFR2.SP.3m.S14", "TNFR2.SP.3m.S15", "TNFR2.SP.3m.S16", "TNFR2.SP.12m.S17", "TNFR2.SP.12m.S18", "TNFR2.SP.12m.S19", "TNFR2.SP.18m.S20", "TNFR2.SP.18m.S21", "TNFR2.SP.18m.S22"]
folder_names = ["TNFR2.SP.3m.S12"]
df_clones_spleen = pd.DataFrame()
for name in folder_names:
    df_clones1 = pd.read_csv(f'data/agrafiotis2021a__VDJ_RAW/{name}/consensus_annotations.csv', encoding='utf-8')
    df_clones_spleen = pd.concat([df_clones_spleen, df_clones1])


#df_clones_bm = pd.read_csv('data/agrafiotis2021a__VDJ_RAW/TNFR2.BM.3m.S1/filtered_contig_annotations.csv', encoding='utf-8')
#df_clones_spleen = pd.read_csv('data/agrafiotis2021a__VDJ_RAW/TNFR2.SP.3m.S12/filtered_contig_annotations.csv', encoding='utf-8')

df_clones_bm = df_clones_bm[df_clones_bm['chain'] == "IGH"]
df_clones_spleen = df_clones_spleen[df_clones_spleen['chain'] == "IGH"]

#select only the first 200 rows
#df_clones_bm = df_clones_bm.head(500)
#df_clones_spleen = df_clones_spleen.head(500)

#combine the columns fwr1, cdr1, fwr2, cdr2, fwr3, cdr3, fwr4 into one column called sequence
df_clones_bm["sequence"] = df_clones_bm["fwr1"] + df_clones_bm["cdr1"] + df_clones_bm["fwr2"] + df_clones_bm["cdr2"] + df_clones_bm["fwr3"] + df_clones_bm["cdr3"] + df_clones_bm["fwr4"]
df_clones_spleen["sequence"] = df_clones_spleen["fwr1"] + df_clones_spleen["cdr1"] + df_clones_spleen["fwr2"] + df_clones_spleen["cdr2"] + df_clones_spleen["fwr3"] + df_clones_spleen["cdr3"] + df_clones_spleen["fwr4"]

#add a column called organ
df_clones_bm["organ"] = "BM"
df_clones_spleen["organ"] = "Spleen"

# Concatenate the dataframes
df_combined = pd.concat([df_clones_bm, df_clones_spleen])

#make the classes balanced by taking the same number of sequences from each organ, the number of sequences is the minimum of the two
min_number = min(len(df_clones_bm), len(df_clones_spleen))
df_clones_bm = df_clones_bm.head(min_number)
df_clones_spleen = df_clones_spleen.head(min_number)


#combine the two dataframes
df = pd.concat([df_clones_bm, df_clones_spleen])

#embed the heavy chain sequences
#model_name = "trained_model_normal_size_hundred_milliondata/cdr3_all_species"
model_name = "30million_mixed_model"
pt_model = RobertaModel.from_pretrained(model_name)

embeddings = get_embeddings(df["sequence"], tokenizer, pt_model)

# Reduce the dimensionality
reducer = umap.UMAP()
scaled_embeddings = StandardScaler().fit_transform(embeddings)
reduced_embeddings = reducer.fit_transform(scaled_embeddings)

#plot the embeddings
plot_umap_by_organ(reduced_embeddings, df,  "results/10million")
plot_umap_by_Vgene(reduced_embeddings, df,  "results/10million")


