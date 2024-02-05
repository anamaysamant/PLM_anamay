import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner

import os 
import sys
import argparse

if "../src" not in sys.path:
        sys.path.append("../src")

from antiberty_model import Antiberty
from ablang_model import Ablang
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert
from tqdm import tqdm


def unique_id_gen(series):
    return list(series)[0]


def compute_evo_velocity(sequence_1, sequence_2, model):
    
    prob_mat_1 = model.calc_probability_matrix(sequence_1).reset_index(drop=True)
    prob_mat_2 = model.calc_probability_matrix(sequence_2).reset_index(drop=True)

    aligner = PairwiseAligner() 
    aligner.extend_gap_score = -0.1
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -4

    sequence_1 = sequence_1.replace("*","")
    sequence_2 = sequence_2.replace("*","")

    alignment = aligner.align(sequence_1,sequence_2)[0]
    alignment_pos = alignment.aligned

    ranges_1 = alignment_pos[0,:,:]
    ranges_2 = alignment_pos[1,:,:]

    count = 0
    evo_velo = 0

    for i in range(ranges_1.shape[0]):
        start_1 = ranges_1[i,0]
        start_2 = ranges_2[i,0]
        subalign_len = ranges_1[i,1] - start_1

        for j in range(subalign_len):

            pos_1 = start_1 + j
            pos_2 = start_2 + j   

            amino_acid_1  = sequence_1[pos_1]
            amino_acid_2  = sequence_2[pos_2]


            if amino_acid_1 != amino_acid_2:

                evo_velo += (prob_mat_1.loc[pos_1,amino_acid_2] - prob_mat_2.loc[pos_2,amino_acid_1])
                count += 1

    if count == 0:
        return 0
    else:
        evo_velo /= count
        return evo_velo

if __name__ == "__main__":

    if "../src" not in sys.path:
        sys.path.append("../src")

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset')      
    parser.add_argument('--mode') 

    args = parser.parse_args()

    dataset = args.dataset
    # mode = args.mode
   
    model_classes = [Ablang, ProtBert, Sapiens]
    model_names = ["ablang","protbert","sapiens"]
    data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))


    for i, model_class in enumerate(model_classes):

        if model_names[i] == "ablang":
            model = Ablang(chain="heavy") 

        elif model_names[i] == "sapiens":
            model = Sapiens(chain_type="H")

        else:
            model = model_class()
        
        
        save_path = os.path.join("..","..","..","data",dataset,f"evo_velo_matrices_{model_names[i]}")

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # output_dict = {}

        for sample in pd.unique(data["sample_id"]):
            
            data_sample = data.loc[data["sample_id"] == sample,:]
            clonotypes = pd.unique(data_sample["clonotype_id_10x"])

            for clonotype in tqdm(clonotypes):
                try:
                    matrix_path = os.path.join(save_path,f"{sample}_{clonotype}_evo_velo.csv")

                    if os.path.exists(matrix_path):
                        continue


                    
                    data_sample_clonotype = data_sample.loc[data_sample["clonotype_id_10x"] == clonotype, : ].reset_index(drop=True)

                    germline_seq = data_sample_clonotype["VDJ_ref.aa"][0].replace("-","")

                    data_sample_clonotype = data_sample_clonotype.groupby("full_VDJ_aa")["barcode"].agg(lambda x: unique_id_gen(x)).reset_index()

                    data_sample_clonotype = data_sample_clonotype.to_dict(orient="records")
                    data_sample_clonotype.append({"full_VDJ_aa":germline_seq,"barcode":f"germline"})
                    data_sample_clonotype = pd.DataFrame(data_sample_clonotype)

                    evo_velo_matrix = pd.DataFrame(0,index=data_sample_clonotype["barcode"], columns=data_sample_clonotype["barcode"])

                    for i in range(data_sample_clonotype.shape[0]):
                        for j in range(i+1,data_sample_clonotype.shape[0]):
                            evo_velo_matrix.iloc[i,j] = compute_evo_velocity(data_sample_clonotype["full_VDJ_aa"][i],
                                                                        data_sample_clonotype["full_VDJ_aa"][j], model=model)

                            evo_velo_matrix.iloc[j,i] = -evo_velo_matrix.iloc[i,j]


                    evo_velo_matrix.to_csv(matrix_path, index=True)
                except:
                        continue
                # output_dict.update({f"{sample}_{clonotype}" : evo_velo_matrix})

        # output_dicts_list.append(output_dict)
        # del(output_dict)
