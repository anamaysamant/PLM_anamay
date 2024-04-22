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
parser.add_argument('--unrecombined', action="store_true") 

args = parser.parse_args()

dataset = args.dataset
unrecombined = args.unrecombined
# mode = args.mode


init_list = [Ablang,Sapiens,ProtBert]
suffixes = ["ablang","sapiens","protbert"]

if unrecombined:
    data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vgene_germline_sequences.csv"))
    data["VDJ_ref.aa"] = data["VDJ_ref.aa"].apply(lambda x: x.replace(".",""))
    save_path = os.path.join("..","..","..","data",dataset,"unrecombined_germline_embeddings")
else:
    data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))
    save_path = os.path.join("..","..","..","data",dataset,"all_germline_embeddings")

columns_to_save = ["barcode","sample_id","chain","v_gene","d_gene","j_gene","c_gene","raw_clonotype_id","VDJ_ref.aa"]
columns_to_unduplicate = ["c_gene","raw_clonotype_id","VDJ_ref.aa"]



data = data.drop_duplicates(columns_to_unduplicate)

if not os.path.isdir(save_path):
    os.mkdir(save_path)

starts = pd.Series([0]*data.shape[0])
ends = data["VDJ_ref.aa"].apply(len)

    
for i,model in enumerate(init_list):

    save_filepath = os.path.join(save_path,f"all_germline_embeddings_{suffixes[i]}.csv.gzip")

    if os.path.exists(save_filepath):
        continue
    
    

    if suffixes[i] in ["ablang","sapiens"]:
        if suffixes[i] == "ablang":
            embeds = Ablang(chain="heavy").fit_transform(sequences=list(data["VDJ_ref.aa"]),
                                                starts=list(starts),ends=list(ends),
                                                path = save_filepath)
            
            embeds = pd.concat([data.loc[:,columns_to_save].reset_index(drop=True),embeds],axis=1)
            
            
        if suffixes[i] == "sapiens":
            embeds = Sapiens(chain_type="H").fit_transform(sequences=(data["VDJ_ref.aa"]),
                                                starts=list(starts),ends=list(ends),
                                                path = save_filepath)

            embeds = pd.concat([data.loc[:,columns_to_save].reset_index(drop=True),embeds],axis=1)

    else:
        model = model()
        embeds = model.fit_transform(sequences=list(data["VDJ_ref.aa"]),starts=list(starts),ends=list(ends))
        embeds = pd.concat([data.loc[:,columns_to_save].reset_index(drop=True),embeds],axis=1)

    embeds.to_csv(save_filepath, index=False, compression="gzip")