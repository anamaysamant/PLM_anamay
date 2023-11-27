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

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='calculates PLL per PLM per sample'
                    )

parser.add_argument('-d','--dataset')           # positional argument
parser.add_argument('--mode') 

args = parser.parse_args()

dataset = args.dataset
mode = args.mode


init_list = [Ablang,ProtBert,Sapiens]
suffixes = ["ablang","protbert","sapiens"]
cellranger_paths = []

data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")

columns_to_save = ["barcode","contig_id","chain","v_gene","d_gene","j_gene","c_gene","raw_clonotype_id","raw_consensus_id","evo_likelihood"]

for sample in os.listdir(data_folder_path):

    cellranger_path = os.path.join(data_folder_path, sample)
    # cellranger_path = os.path.join(cellranger_path, os.listdir(cellranger_path)[0])

    if not (os.path.isdir(os.path.join(cellranger_path,"evo_likelihoods"))):
        os.mkdir(os.path.join(cellranger_path,"evo_likelihoods"))

    evo_folder = os.path.join(cellranger_path,"evo_likelihoods")
    repertoire_file_path = os.path.join(cellranger_path,"filtered_contig_annotations.csv")

    repertoire_file = pd.read_csv(repertoire_file_path)
    if (mode == "cdr3_only"):
        repertoire_file["full_sequence"] = repertoire_file["cdr3"]

        if not (os.path.isdir(os.path.join(evo_folder,"cdr3_only"))):
            os.mkdir(os.path.join(evo_folder,"cdr3_only"))

        save_path = os.path.join(evo_folder,"cdr3_only")
    elif (mode == "full_VDJ" or mode == "cdr3_from_VDJ"):
        repertoire_file["full_sequence"] = repertoire_file["fwr1"] + repertoire_file["cdr1"] + repertoire_file["fwr2"] + \
                                        repertoire_file["cdr2"] + repertoire_file["fwr3"] + repertoire_file["cdr3"] + repertoire_file["fwr4"]
        
        if not (os.path.isdir(os.path.join(evo_folder,"full_VDJ"))):
            os.mkdir(os.path.join(evo_folder,"full_VDJ"))

        save_path = os.path.join(evo_folder,"full_VDJ")
    
    if mode == "cdr3_from_VDJ":
        x = repertoire_file["fwr1"] + repertoire_file["cdr1"] + repertoire_file["fwr2"] + \
                                        repertoire_file["cdr2"] + repertoire_file["fwr3"]

        starts = x.apply(lambda x: len(x) if not pd.isna(x) else None)

        y = repertoire_file["fwr1"] + repertoire_file["cdr1"] + repertoire_file["fwr2"] + \
                                        repertoire_file["cdr2"] + repertoire_file["fwr3"] + repertoire_file["cdr3"]  
         
        ends = y.apply(lambda x: len(x) if not pd.isna(x) else None)

        if not (os.path.isdir(os.path.join(evo_folder,"cdr3_from_VDJ"))):
            os.mkdir(os.path.join(evo_folder,"cdr3_from_VDJ"))
        
        save_path = os.path.join(evo_folder,"cdr3_from_VDJ")

    else:
        starts = pd.Series([0]*repertoire_file.shape[0])
        ends = repertoire_file["full_sequence"].apply(len)
    
    for i,model in enumerate(init_list):
        if os.path.exists(os.path.join(save_path,f"evo_likelihood_{suffixes[i]}.csv")):
            continue
        
        if suffixes[i] in ["ablang","sapiens"]:
            repertoire_file["evo_likelihood"] = "dummy"
            is_heavy_chain = list(repertoire_file["chain"] == "IGH")
            is_light_chain = list(repertoire_file["chain"] != "IGH")
            if suffixes[i] == "ablang":
                repertoire_file.loc[is_heavy_chain,"evo_likelihood"] = Ablang(chain="heavy").calc_pseudo_likelihood_sequence(list(repertoire_file[is_heavy_chain]["full_sequence"]),list(starts[is_heavy_chain]),list(ends[is_heavy_chain]))
                repertoire_file.loc[is_light_chain,"evo_likelihood"] = Ablang(chain="light").calc_pseudo_likelihood_sequence(list(repertoire_file[is_light_chain]["full_sequence"]),list(starts[is_light_chain]),list(ends[is_light_chain]))
            if suffixes[i] == "sapiens":
                repertoire_file.loc[is_heavy_chain,"evo_likelihood"] = Sapiens(chain_type="H").calc_pseudo_likelihood_sequence(list(repertoire_file[is_heavy_chain]["full_sequence"]),list(starts[is_heavy_chain]),list(ends[is_heavy_chain]))
                repertoire_file.loc[is_light_chain,"evo_likelihood"] = Sapiens(chain_type="L").calc_pseudo_likelihood_sequence(list(repertoire_file[is_light_chain]["full_sequence"]),list(starts[is_light_chain]),list(ends[is_light_chain]))

        else:
            model = model()
            repertoire_file["evo_likelihood"] = model.calc_pseudo_likelihood_sequence(list(repertoire_file["full_sequence"]),list(starts),list(ends))

        repertoire_file[columns_to_save].to_csv(os.path.join(save_path,f"evo_likelihood_{suffixes[i]}.csv"), index=False)
