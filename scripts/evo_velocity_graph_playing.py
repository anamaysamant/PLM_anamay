import numpy as np
import pandas as pd
import os
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap as umap
from sklearn.decomposition import KernelPCA

IgG_subtypes = ["IGHG1","IGHG2B","IGHG2C","IGHG3"]

embeddings_file = pd.read_csv("../../data/OVA_mouse/VDJ/S1/embeddings/full_VDJ/embeddings_sapiens.csv.gzip", compression="gzip")
embeddings_file = embeddings_file.loc[embeddings_file["chain"] == "IGH",:].reset_index(drop=True)
embeddings_file["c_gene"] = embeddings_file["c_gene"].replace(IgG_subtypes,"IGHG")
embeddings_file["v_gene_family"] = embeddings_file["v_gene"].apply(lambda x: x.split('-')[0])

embedding_cols = [col for col in list(embeddings_file.columns) if col.startswith("dim")]
metadata_cols = list(set(embeddings_file.columns) - set(embedding_cols))

embeddings_file = embeddings_file.drop_duplicates(embedding_cols).reset_index(drop=True)
embeddings_file = embeddings_file.dropna(subset=embedding_cols)

only_embeddings = embeddings_file[embedding_cols].copy()
metadata = embeddings_file[metadata_cols].copy()

reducer = umap.UMAP(random_state=0, n_jobs=1,n_neighbors=50,min_dist=0.5)

proj = reducer.fit_transform(only_embeddings)
proj = pd.DataFrame(proj)
proj = pd.concat([metadata,proj], axis=1)
proj = pd.DataFrame(proj.iloc[:,-2:])

proj.columns = ["dim_0","dim_1"]

max_silhouette_score = -np.Inf
best_k = None

# dim_reduced_embeddings = KernelPCA(n_components=2, kernel='rbf').fit_transform(only_embeddings)



for k in range(2,20):

    labels = AgglomerativeClustering(n_clusters=k).fit_predict(only_embeddings)
    sil_score = adjusted_rand_score(labels_true = metadata["v_gene"], labels_pred=labels)
    print(sil_score)

    if (sil_score > max_silhouette_score):
        max_silhouette_score = sil_score
        best_k = k



# metadata.iloc[np.where(kNN_graph[60,:] == 1)[0], :]


