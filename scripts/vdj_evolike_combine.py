import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset')

args = parser.parse_args()

dataset = args.dataset
vdj = pd.read_csv(f"../../../data/{dataset}/all_samples_VDJ_germline_gaps.csv", sep=",")
  
evo_likelihood = pd.read_csv(f"../../../data/{dataset}/all_samples_evo_likelihoods/{dataset}_all_samples_all_plms_all_sources_evo_likelihoods.csv")

IgG_subtypes = ["IGHG1","IGHG2","IGHG2B","IGHG2C","IGHG3","IGHG4"]
IgA_subtypes = ["IGHA1","IGHA2"]

vdj = vdj.loc[vdj["Nr_of_VDJ_chains"] == 1, :]
evo_likelihood = evo_likelihood.loc[evo_likelihood["chain"] == "IGH",:]
  
evo_likelihood["barcode"] = evo_likelihood["barcode"].apply(lambda x: x.split("-")[0])
  
join = pd.merge(vdj, evo_likelihood)
join = join.sort_values("sample_id")

join["VDJ_cgene"] = join["VDJ_cgene"].replace(IgG_subtypes,"IGHG").replace(IgA_subtypes,"IGHA")
join["c_gene"] = join["VDJ_cgene"]
join["v_gene_family"] = join["v_gene"].apply(lambda x: x.split('-')[0])
join["VDJ_cdr3_length"] = join["VDJ_cdr3s_aa"].apply(len)
join["full_VDJ_aa"] = join.apply(lambda x: x["HC_fwr1"]+ x["HC_cdr1"]+ x["HC_fwr2"]+ x["HC_cdr2"]+ x["HC_fwr3"]+ x["HC_cdr3"]+ x["HC_fwr4"], axis=1)
join["full_VDJ_length"] = join["full_VDJ_aa"].apply(len)
  
join.to_csv(f"../../../data/{dataset}/vdj_evolike_combine.csv", sep=",", index=False)