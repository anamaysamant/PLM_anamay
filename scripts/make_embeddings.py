import sys
import os
sys.path.append(os.getcwd()+"/src")

from sapiens_model import Sapiens
from protbert import ProtBert
from antiberty_model import Antiberty
from ablang_model import Ablang
from utils import prepare_sequence
from ESM_model import ESM
from ESM_prob_model import ESM_prob
from TCRBert_model import TCRBert
import pandas as pd
import yaml

#Load settings, data and for each method check for the provided arguements
with open("config.yml", "r") as f:
    settings = yaml.safe_load(f)

if settings["model"] == 'Ablang':

    print("Using Ablang")
    if "method" not in settings:
        settings["method"] = "seqcoding"
    
    if "chain" in settings:
        chain = settings["chain"]
        
        print("Using {} chain".format(chain))
        model = Ablang(chain,file_name=sys.argv[3], method = settings["method"])
    else:
        print("Using default chain")
        model = Ablang(file_name=sys.argv[3], method = settings["method"])
    
    data = pd.read_csv(sys.argv[1])["sequence"]

elif settings["model"] == 'Protbert':

    print("Using Protbert")
    data = prepare_sequence(sys.argv[1])

    if "method" in settings:
        model = ProtBert(settings["method"], file_name=sys.argv[3])
    else:
        model = ProtBert(file_name=sys.argv[3])

elif settings["model"] == "ESM":
    print("Using ESM")
    if "method" in settings:
        if settings["method"] == "prob":
            print("Likelihoods mode")
            data = pd.read_csv(sys.argv[1])[sys.argv[2]]
            model = ESM_prob(file_name=sys.argv[3])
        else:
            print("Embeddings mode")
            data = prepare_sequence(sys.argv[1])
            model = ESM(settings["method"], file_name=sys.argv[3])
    else:
        data = prepare_sequence(sys.argv[1])
        model = ESM(file_name=sys.argv[3])

elif settings["model"] == "Antiberty":

    print("Using Antiberty")
    data = pd.read_csv(sys.argv[1])["sequence"]

    if "token" in settings:
        model = Antiberty(settings["token"], file_name=sys.argv[3])
    else:
        model = Antiberty(file_name=sys.argv[3])

elif settings["model"] == "Sapiens":
    if "chain" not in settings:

        settings["chain"] = "H"
    if "method" not in settings:
        
        settings["method"] = "average"
    
    data = pd.read_csv(sys.argv[1])[sys.argv[2]]
    model = Sapiens(settings["chain"], settings["method"], file_name=sys.argv[3])
    

elif settings["model"] == 'TCRBert':
    print("Using TCRBERT")
    data = pd.read_csv(sys.argv[1])['sequence']
    if "method" in settings:
        model = TCRBert(settings["method"],file_name=sys.argv[3])
    else:
        model = TCRBert("average",file_name=sys.argv[3])

model.fit_transform(sequences = data)
