#Inference functions
import torch
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
import matplotlib.patches as mpatches


from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def get_embeddings(input_seqs, tokenizer, pt_model):
    input_seqs = input_seqs.tolist()
    embeddings = []

    #k = 0 use this if you want to only work with a subset of the data
    for seq in input_seqs:
        #if k < 1000:
            tokenized_input = tokenizer(seq, return_tensors='pt', padding=True)
            if tokenized_input["input_ids"].shape[1] > 150:
                df = df.drop(df.index[k])
                #k = k + 1
                continue
            output = pt_model(**tokenized_input)
            embedding = torch.mean(output.last_hidden_state, axis = 1)[0]

            embedding = embedding.detach()
            embeddings.append(embedding)

    all_embeddings = np.zeros((len(embeddings), 768)) 
    n = 0 
    for embedding in embeddings:
        embedding = np.array(embedding)
        all_embeddings[n,:] = embedding.tolist()
        n = n + 1

    return all_embeddings

def plot_umap_by_Vgene(reduced_embeddings, df, results_path):
    # List of classes
    #classes = ['IGHV5-16', 'IGKV1-88', 'IGHV11-2', 'IGKV14-126', ...]  # shortened for brevity

    # List of classes
    classes = [
    'IGHV5-16', 'IGKV1-88', 'IGHV11-2', 'IGKV14-126', 'IGHV1-76', 'IGKV3-4',
    'IGHV1-18', 'IGKV12-44', 'IGHV4-1', 'IGKV4-74', 'IGHV5-17', 'IGKV8-24',
    'IGHV7-1', 'IGKV15-103', 'IGHV6-3', 'IGKV5-39', 'IGHV1-19', 'IGKV6-23',
    'IGHV2-6', 'IGKV1-117', 'IGHV1-55', 'IGKV14-111', 'IGHV3-6', 'IGKV1-110',
    'IGHV1-72', 'IGKV19-93', 'IGHV2-5', 'IGHV8-12', 'IGKV4-72', 'IGKV4-91',
    'IGHV9-3', 'IGKV4-61', 'IGHV1-80', 'IGKV4-57', 'IGHV1-26', 'IGHV5-6',
    'IGKV6-15', 'IGHV10-3', 'IGKV10-96', 'IGHV1-49', 'IGKV6-25', 'IGHV1-53',
    'IGHV1-9', 'IGKV2-137', 'IGHV6-6', 'IGKV4-69', 'IGHV1-15', 'IGKV8-30',
    'IGKV5-43', 'IGKV10-94', 'IGHV1-78', 'IGLV1', 'IGHV5-15', 'IGKV3-7',
    'IGHV1-74', 'IGKV9-124', 'IGKV12-89', 'IGKV8-27', 'IGHV1-62-2', 'IGKV2-109',
    'IGKV8-21', 'IGHV5-4', 'IGKV8-28', 'IGKV4-53', 'IGHV1-75', 'IGLV3',
    'IGKV3-2', 'IGKV12-46', 'IGHV5-9-1', 'IGHV1-7', 'IGHV14-4', 'IGHV9-1',
    'IGKV4-50', 'IGKV4-90', 'IGKV4-55', 'IGHV1-82', 'IGLV2', 'IGHV10-1',
    'IGKV1-135', 'IGHV1-5', 'IGKV3-12', 'IGKV4-86', 'IGHV1-71', 'IGKV17-127',
    'IGHV13-2', 'IGKV3-10', 'IGKV4-59', 'IGHV1-22', 'IGKV17-121', 'IGKV6-20',
    'IGHV1-63', 'IGKV3-5', 'IGHV14-2', 'IGHV14-3', 'IGHV1-39', 'IGKV13-85',
    'IGHV2-2', 'IGKV4-80', 'IGHV2-3', 'IGHV8-8', 'IGHV2-4', 'IGHV7-3',
    'IGHV2-9', 'IGKV6-17', 'IGKV5-45', 'IGHV3-1', 'IGKV5-48', 'IGKV6-32',
    'IGKV4-70', 'IGHV1-69', 'IGKV9-120', 'IGHV1-81', 'IGKV8-34', 'IGHV9-4',
    'IGHV1-42', 'IGKV13-84', 'IGHV5-2', 'IGHV2-9-1', 'IGHV12-3', 'IGKV8-19',
    'IGHV1-64', 'IGHV5-12', 'IGHV8-5', 'IGHV1-66', 'IGHV1-34', 'IGHV1-50',
    'IGHV1-12', 'IGHV1-61', 'IGKV4-79', 'IGHV3-4', 'IGHV14-1', 'IGHV1-54',
    'IGHV1-47', 'IGKV1-122', 'IGKV2-112', 'IGHV1-84', 'IGHV1-20', 'IGHV1-52',
    'IGHV1-11', 'IGKV8-16', 'IGHV1-77', 'IGHV3-5', 'IGHV1-85', 'IGKV14-100',
    'IGKV3-1', 'IGHV1-58', 'IGKV16-104', 'IGHV1-37', 'IGHV1-59', 'IGKV4-63',
    'IGKV3-9', 'IGKV4-71', 'IGKV12-98', 'IGKV1-99', 'IGHV7-4', 'IGKV4-68',
    'IGKV6-13', 'IGHV1-4', 'IGHV1-36', 'IGKV4-58', 'IGKV4-57-1', 'IGKV1-133',
    'IGKV12-41', 'IGKV14-130', 'IGKV6-14', 'IGKV12-38', 'IGHV9-2', 'IGHV1-31',
    'IGKV8-18', 'IGKV4-54', 'IGKV11-125', 'IGHV2-6-8', 'IGHV1-2', 'IGHV1-56',
    'IGHV11-1', 'IGKV9-129', 'IGHV5-9', 'IGKV9-123', 'IGKV4-81', 'IGHV3-3',
    'IGKV4-56', 'IGKV6-29', 'IGKV5-37', 'IGKV1-132', 'IGHV8-6', 'IGHV8-4',
    'IGHV15-2', 'IGKV4-51', 'IGKV4-78', 'IGHV1-23', 'IGKV3-3', 'IGHV1-43',
    'IGKV20-101-2'
    ]

    # Clean up the classes by ignoring numbers after the dash
    cleaned_classes = list(set([item.split('-')[0] for item in classes]))

    # Select a color palette from matplotlib
    colors = plt.cm.tab20c(np.linspace(0, 1, len(cleaned_classes)))

    # Create a dictionary mapping each class to a color
    color_map = {cleaned_classes[i]: colors[i] for i in range(len(cleaned_classes))}

    # Make a list of colors for each example corresponding to its class, works for files with two different naming conventions
    if "V_CALL" not in df.columns:
        df["v_gene"] = df["v_gene"].str.split('-').str.get(0)
        colors_v = df["v_gene"].map(color_map)
    else:
        df["V_CALL"] = df["V_CALL"].str.split('-').str.get(0)
        colors_v = df["V_CALL"].map(color_map)

    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.7,
        s = 3,
        c = colors_v
    )

    unique_classes = df["v_gene"].unique() if "V_CALL" not in df.columns else df["V_CALL"].unique()

    # Create custom legend patches for the unique classes
    legend_patches = [mpatches.Patch(color=color_map[class_], label=class_) for class_ in unique_classes if class_ in color_map]
    
    # Add the legend to the plot
    plt.legend(handles=legend_patches, loc='best', fontsize='small')

    plt.title("UMAP projection of the embeddings colored by V gene usage")

    plt.savefig(f"{results_path}/umap_V.pdf")
    plt.show()

def plot_umap_by_mutations(reduced_embeddings, df, results_path):
    # Generate a range of blue colors
    start_color = np.array([160, 160, 240])  # RGB color code for pure blue
    end_color = np.array([100, 30, 30])  # RGB color code for light blue

    # Create a color vector based on the numbers in the specified column
    #min_value = df["MUT_TOTAL"][0:1000].min()
    #max_value = df["MUT_TOTAL"][0:1000].max()
    min_value = 0
    max_value = 20

    # Calculate the normalized values (between 0 and 1) for each number
    normalized_values = (df["MUT_TOTAL"] - min_value) / (max_value - min_value)
    for i in range(len(normalized_values)):
        if normalized_values[i] > 1:
            normalized_values[i] = 1

    # Interpolate between the start_color and end_color based on the normalized values
    colors_mutations = [tuple((start_color + normalized * (end_color - start_color)) / 255.0)
          for normalized in normalized_values]
    
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.7,
        s = 3,
        c = colors_mutations
    )
    plt.title("UMAP projection of the embeddings colored by mutation count")

    plt.savefig(f"{results_path}/umap_mutations.pdf")
    plt.show()
    
def plot_umap_by_memory(reduced_embeddings, df, results_path):
    color_map = {
        'Mem': 'red',
        'Naive': 'blue',
    }
    colors_mem_naive = df["Subset"].map(color_map).fillna("cyan")

    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.7,
        s = 3,
        c = colors_mem_naive
    )
    plt.title("UMAP projection of the embeddings colored by memory/naive status")

    plt.savefig(f"{results_path}/umap_memory.pdf")
    plt.show()

def plot_umap_by_organ(reduced_embeddings, df, results_path):
    color_map = {
        'BM': 'red',
        'Spleen': 'blue',
    }
    colors_organ = df["organ"].map(color_map).fillna("red")

    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.4,
        s = 2.5,
        c = colors_organ
    )

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=organ, markersize=10, markerfacecolor=color_map[organ]) for organ in color_map]
    plt.legend(handles=handles)

    plt.title("UMAP projection of the embeddings colored by organ")
    plt.savefig(f"{results_path}/umap_organ.pdf")
    plt.show()
     



