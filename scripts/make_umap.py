import umap.umap_ as umap
import pandas as pd
import plotly.express as px
import sys
import yaml

umap_data = pd.read_csv("outfiles/{}/embeddings.csv".format(sys.argv[1]), index_col=0)
with open("config.yml", "r") as f:
    settings = yaml.safe_load(f)
if "data_path" in settings and "VGM" not in settings["mode"]:
    metadata = pd.read_csv(settings["data_path"])
else:
    umap_data = pd.read_csv("outfiles/{}/clones.csv".format(sys.argv[1]), index_col=0)
reducer = umap.UMAP(random_state=0)
umap_data_inv = umap_data

proj = reducer.fit_transform(umap_data_inv)
proj = pd.DataFrame(proj).reset_index()
metadata = metadata.reset_index()
total = pd.merge(pd.DataFrame(proj), metadata, how = "inner")

fig_2d = px.scatter(
    total, x=0, y=1, color = total["sample"].astype(str)
)

fig_2d.update_layout(
    plot_bgcolor='white',
    showlegend = False
)
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
fig_2d.write_image("outfiles/{}/UMAP.pdf".format(sys.argv[1]))