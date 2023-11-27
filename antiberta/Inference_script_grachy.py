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

#################################

#get embedding

model_name = "antiberta_large"
pt_model = RobertaModel.from_pretrained(model_name)

df = pd.read_csv('data/random_graichy_1000.csv', encoding='utf-8')

print(df.head())

#pd.DataFrame(all_embeddings).to_csv("embeddings.csv")

embedding = get_embeddings(df["SEQUENCE_IMGT"], tokenizer, pt_model)

#pd.read_csv("embeddings.csv")

# Reduce the dimensionality
reducer = umap.UMAP()
scaled_embeddings = StandardScaler().fit_transform(embedding)
reduced_embeddings = reducer.fit_transform(scaled_embeddings)

plot_umap_by_Vgene(reduced_embeddings, df, "graichy_antiberta_large")
plot_umap_by_mutations(reduced_embeddings, df, "graichy_antiberta_large")
plot_umap_by_memory(reduced_embeddings, df, "graichy_antiberta_large")