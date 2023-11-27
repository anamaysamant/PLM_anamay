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

#read in bm data from the different folders (either all folders or just one folder to do it by sample)
#folder_names = ["TNFR2.BM.3m.S1", "TNFR2.BM.3m.S2", "TNFR2.BM.3m.S3", "TNFR2.BM.3m.S4", "TNFR2.BM.3m.S5", "TNFR2.BM.12m.S6", "TNFR2.BM.12m.S7", "TNFR2.BM.12m.S8", "TNFR2.BM.18m.S9", "TNFR2.BM.18m.S10", "TNFR2.BM.18m.S11"]
folder_names = ["TNFR2.BM.3m.S1"]
df_bm = pd.DataFrame()
for name in folder_names:
    df_bm1 = pd.read_csv(f'data/agrafiotis2021a__VDJ_RAW/{name}/filtered_contig_annotations.csv', encoding='utf-8')
    df_bm = pd.concat([df_bm, df_bm1])


#read in spleen data from the different folders
#folder_names = ["TNFR2.SP.3m.S12", "TNFR2.SP.3m.S13", "TNFR2.SP.3m.S14", "TNFR2.SP.3m.S15", "TNFR2.SP.3m.S16", "TNFR2.SP.12m.S17", "TNFR2.SP.12m.S18", "TNFR2.SP.12m.S19", "TNFR2.SP.18m.S20", "TNFR2.SP.18m.S21", "TNFR2.SP.18m.S22"]
folder_names = ["TNFR2.SP.3m.S12"]
df_spleen = pd.DataFrame()
for name in folder_names:
    df_spleen1 = pd.read_csv(f'data/agrafiotis2021a__VDJ_RAW/{name}/filtered_contig_annotations.csv', encoding='utf-8')
    df_spleen = pd.concat([df_spleen, df_spleen1])


#df_bm = pd.read_csv('data/agrafiotis2021a__VDJ_RAW/TNFR2.BM.3m.S1/filtered_contig_annotations.csv', encoding='utf-8')
#df_spleen = pd.read_csv('data/agrafiotis2021a__VDJ_RAW/TNFR2.SP.3m.S12/filtered_contig_annotations.csv', encoding='utf-8')

df_bm = df_bm[df_bm['chain'] == "IGK"]
df_spleen = df_spleen[df_spleen['chain'] == "IGK"]

#select only the first 200 rows
df_bm = df_bm.head(500)
df_spleen = df_spleen.head(500)

#combine the columns fwr1, cdr1, fwr2, cdr2, fwr3, cdr3, fwr4 into one column called sequence
df_bm["sequence"] = df_bm["fwr1"] + df_bm["cdr1"] + df_bm["fwr2"] + df_bm["cdr2"] + df_bm["fwr3"] + df_bm["cdr3"] + df_bm["fwr4"]
df_spleen["sequence"] = df_spleen["fwr1"] + df_spleen["cdr1"] + df_spleen["fwr2"] + df_spleen["cdr2"] + df_spleen["fwr3"] + df_spleen["cdr3"] + df_spleen["fwr4"]

#add a column called organ
df_bm["organ"] = "BM"
df_spleen["organ"] = "Spleen"

#combine the two dataframes
df = pd.concat([df_bm, df_spleen])

#embed the heavy chain sequences
model_name = "/Users/surf/Documents/Thesis/antiberta/output_1_million_model_mouse_light/mouseB_light"
pt_model = RobertaModel.from_pretrained(model_name)

embeddings = get_embeddings(df["cdr3"], tokenizer, pt_model)

# Reduce the dimensionality
reducer = umap.UMAP()
scaled_embeddings = StandardScaler().fit_transform(embeddings)
reduced_embeddings = reducer.fit_transform(scaled_embeddings)

#plot the embeddings
plot_umap_by_organ(reduced_embeddings, df,  "results_agrafiotis")