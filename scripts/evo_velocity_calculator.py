import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner

import os 
import sys

if r"C:/Users/anama/OneDrive/Desktop/Anamay_Thesis/Python/PLM/src" not in sys.path:
    sys.path.append(r"C:/Users/anama/OneDrive/Desktop/Anamay_Thesis/Python/PLM/src")

os.chdir(r"C:/Users/anama/OneDrive/Desktop/Anamay_Thesis/Python/PLM/scripts")


from antiberty_model import Antiberty
from ablang_model import Ablang
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert
from tqdm import tqdm


def unique_id_gen(series):
    return list(series)[0]


def compute_evo_velocity(sequence_1, sequence_2, model):

    prob_mat_1 = model.calc_probability_matrix(sequence_1)
    prob_mat_2 = model.calc_probability_matrix(sequence_2)

    aligner = PairwiseAligner() 
    aligner.extend_gap_score = -0.1
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -4

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
    
    model_class = ProtBert
    model_name = "protbert"
    data = pd.read_csv("../../../data/OVA_mouse/vdj_evolike_combine.csv")

    # output_dicts_list = []

    if model_name == "ablang":
        model = Ablang(chain="heavy") 

    elif model_name == "sapiens":
        model = Sapiens(chain_type="H")

    else:
        model = model_class()
    
    
    output_dict = {}

    for sample in pd.unique(data["sample_id"]):
        
        data_sample = data.loc[data["sample_id"] == sample,:]

        for clonotype in tqdm(pd.unique(data_sample["clonotype_id_10x"])):

            data_sample_clonotype = data_sample.loc[data_sample["clonotype_id_10x"] == clonotype, : ]
            data_sample_clonotype = data_sample_clonotype.groupby("VDJ_sequence_aa")["barcode"].agg(lambda x: unique_id_gen(x)).reset_index()
            
            evo_velo_matrix = pd.DataFrame(0,index=data_sample_clonotype["barcode"], columns=data_sample_clonotype["barcode"])

            for i in range(data_sample_clonotype.shape[0]):
                for j in range(i+1,data_sample_clonotype.shape[0]):
                    try:
                        evo_velo_matrix.iloc[i,j] = compute_evo_velocity(data_sample_clonotype["VDJ_sequence_aa"][i],
                                                                    data_sample_clonotype["VDJ_sequence_aa"][j], model=model)

                        evo_velo_matrix.iloc[j,i] = -evo_velo_matrix.iloc[i,j]
                    except:
                        evo_velo_matrix.iloc[i,j] = None
                        evo_velo_matrix.iloc[i,j] = None


            output_dict.update({f"{sample}_{clonotype}" : evo_velo_matrix})

        # output_dicts_list.append(output_dict)
        # del(output_dict)
