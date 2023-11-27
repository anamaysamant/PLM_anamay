# Inference_script that reads in an embedding (for example from Evgenios pipeline) and plots UMAPs
# Should be rewritten with inference_functions.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
import seaborn as sns
import os
import torch

#get embedding
df = pd.read_csv('data/graichy_nucleotide_sequences_1000_random.csv', encoding='utf-8')

input_seqs = df["SEQUENCE_IMGT"]
input_seqs = input_seqs.tolist()

embedding = pd.read_csv('/Users/surf/Documents/Thesis/EvgeniosKladis/outfiles/embeddings_DNABert.csv')
#turn all NaN values to 0
embedding = embedding.fillna(0)

# Reduce the dimensionality
reducer = umap.UMAP()
scaled_embeddings = StandardScaler().fit_transform(embedding)
reduced_embeddings = reducer.fit_transform(scaled_embeddings)

#colors = np.full(len(scaled_embeddings),'#1f77b4')
#colors[0:125] = "#ff7f0e"

color_map = {
    'IGHV1': 'red',
    'IGHV2': 'blue',
    'IGHV3': 'green',
    'IGHV4': 'orange',
    'IGHV5': 'purple',
    'IGHV6': 'yellow',
    'IGHV7': 'cyan',
}

color_map2 = {
    'Mem': 'red',
    'Naive': 'blue',
}

#generate the color gradient for the mutation count
start_color = np.array([0, 0, 255])  # RGB color code for pure blue
mid_color = np.array([240,255,240])  # RGB color code for white (scaled to 0-1 range)
end_color = np.array([255, 0, 0])  # RGB color code for pure red (scaled to 0-1 range)

# Create a color vector based on the numbers in the specified column
min_value = df["MUT_TOTAL"].min()
max_value = df["MUT_TOTAL"].max()

# Calculate the normalized values (between 0 and 1) for each number
normalized_values = (df["MUT_TOTAL"] - min_value) / (max_value - min_value)

# Calculate the normalized values (between 0 and 1) for each number
min_value = df["MUT_TOTAL"].min()
max_value = df["MUT_TOTAL"].max()
normalized_values = (df["MUT_TOTAL"] - min_value) / (max_value - min_value)

# Interpolate between the colors based on the normalized values
colors_mutations = [
    tuple(
        (
            (1 - normalized) * start_color
            + normalized * mid_color
            if normalized <= 0.5
            else (1 - normalized) * mid_color + (normalized - 0.5) * end_color
        )
        / 255.0
    )
    for normalized in normalized_values
]

# Create a new column containing the corresponding colors based on the values in the original column
colors_v = df["V_CALL"].map(color_map).fillna("red")
colors_v = colors_v[0:1000]
#,colors = colors.fillna("gray", inplace=True)
#colors = colors[0:df.shape[0]]

colors_mem_naive = df["Subset"].map(color_map2).fillna("red")
colors_mem_naive = colors_mem_naive[0:1000]

# Plot the results
plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    alpha=0.7,
    s = 3,
    c = colors_v
)
plt.title("UMAP projection of the embeddings colored by V gene usage")

plt.savefig("umap_V.pdf")
plt.show()

# Plot the results
plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    alpha=0.7,
    s = 3,
    c = colors_mutations
)
plt.title("UMAP projection of the embeddings colored by mutation count")

plt.savefig("umap_mutations.pdf")
plt.show()

# Plot the results
plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    alpha=0.7,
    s = 3,
    c = colors_mem_naive
)
plt.title("UMAP projection of the embeddings colored by memory/naive status")

plt.savefig("umap_memory.pdf")
plt.show()

