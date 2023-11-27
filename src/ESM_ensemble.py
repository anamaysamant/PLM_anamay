from transformers import AutoTokenizer, EsmModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from esm import pretrained
import pickle as pkl
import os
import sys
import scipy

sys.path.append("../scripts")
from utils import get_pseudo_likelihood

class ESM_ensemble():

    """
    Class for the protein Language Model
    """

    def __init__(self, method = "average", file_name = "."):
        
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")

        self.models = []
        self.models.append(EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S"))

        for i in range(1,6):
            self.models.append(EsmModel.from_pretrained(f"facebook/esm1v_t33_650M_UR90S_{i}"))

        self.name_ = "esm1b_t33_650M_UR50S"

        self.method = method
        self.file = file_name
        self.repr_layer_ = -1


        model, alphabet = pretrained.load_model_and_alphabet(self.name_)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
        #model and alphabet
        self.model_ = model
        self.alphabet_ = alphabet
        self.embedding_dim = 1280
        

    def fit_transform(self, sequences:list, batches = 10):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        
        batches: `int`
        Number of batches. Per batch a checkpoint file will be saved
        ------

        None, saved the embeddings in the embeddings.csv
        """
        batch_size = round(len(sequences)/batches)
        print("\nUsing the {} method".format(self.method))
        
        pooler_zero = np.zeros((self.embedding_dim, len(sequences)))
        for sequence,_ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            if not isinstance(sequence[1], float):
                tokenized_sequences = self.tokenizer(sequence[1], return_tensors= 'pt') #return tensors using pytorch
                output = torch.zeros(self.embedding_dim)
                
                for model in self.models:

                    per_model_output = model(**tokenized_sequences)

                    if self.method == "average":
                        per_model_output = torch.mean(per_model_output.last_hidden_state, axis = 1)[0]
                    
                    elif self.method == "pooler":
                        per_model_output = per_model_output.pooler_output[0]
                    
                    elif self.method == "last":
                        per_model_output = per_model_output.last_hidden_state[0,-1,:]

                    elif self.method == "first":
                        per_model_output = per_model_output.last_hidden_state[0,0,:]
                    
                    output += per_model_output
                        
                output /= len(self.models)

                pooler_zero[:,sequence[0]] = output.tolist()
                if sequence[0] % (batch_size+1) == 0:   #Checkpoint save
                    pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv")

        pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv")

    def calc_evo_likelihood_matrix_per_position(self, sequences:list, batch_size = 10):

        batch_converter = self.alphabet_.get_batch_converter()
        data = []
        for i,sequence in enumerate(sequences):
            data.append(("protein{}".format(i),sequence))
        probs = []
        count = 0

        #One sequence at a time
        for sequence,_ in zip(data,tqdm(range(len(data)))):
            #Tokenize & run using the last layer
            _, _, batch_tokens = batch_converter([sequence])
            batch_tokens = batch_tokens.to("cuda:0" if torch.cuda.is_available() else "cpu")
            avg_probs = np.zeros((len(list(sequence)),len(self.alphabet_ + 2)))
            for model in self.models:

                out = model(batch_tokens,repr_layers = [self.repr_layer_],return_contacts = False)
                #Retrieve numerical values for each possible token (including aminoacids and special tokens) in each position
                logits = out["logits"][0].cpu().detach().numpy()
                #Turn them into probabilties 
                prob = scipy.special.softmax(logits,axis = 1)
                avg_probs += prob
            
            avg_probs /= len(self.models)
            #Preprocessing probabilities, removing CLS and SEP tokens and removing probabilities of Special aminoacids and tokens of the model.
            df = pd.DataFrame(avg_probs, columns = self.alphabet_.all_toks)
            df = df.iloc[:,4:-4]
            df = df.loc[:, df.columns.isin(["U","Z","O","B","X"]) == False]
            #removing CLS and SEP
            df = df.iloc[1:-1,:]
            df = df.reindex(sorted(df.columns), axis=1)
            probs.append(df)

            count+=1

        likelihoods = get_pseudo_likelihood(probs, sequences)
        pkl.dump([probs,likelihoods],open("outfiles/"+self.file+"/probabilities_pseudo.pkl","wb"))
        print("done with predictions")

    def calc_pseudo_likelihood_sequence(self, sequences:list, batch_size = 10):

        batch_converter = self.alphabet_.get_batch_converter()
        data = []
        for i,sequence in enumerate(sequences):
            data.append(("protein{}".format(i),sequence))
        pll_all_sequences = []
        #One sequence at a time
        for sequence,_ in zip(data,tqdm(range(len(data)))):
            #Tokenize & run using the last layer
            amino_acids = list(sequence[1])
            _, _, batch_tokens = batch_converter([sequence])
            batch_tokens = batch_tokens.to("cuda:0" if torch.cuda.is_available() else "cpu")
            for model in self.models:

                avg_probs = np.zeros((len(list(sequence)),len(self.alphabet_ + 2)))
                for model in self.models:

                    out = model(batch_tokens,repr_layers = [self.repr_layer_],return_contacts = False)
                    #Retrieve numerical values for each possible token (including aminoacids and special tokens) in each position
                    logits = out["logits"][0].cpu().detach().numpy()
                    #Turn them into probabilties 
                    prob = scipy.special.softmax(logits,axis = 1)
                    avg_probs += prob
            
            avg_probs /= len(self.models)
            df = pd.DataFrame(avg_probs, columns = self.alphabet_.all_toks)

            per_position_ll = []
            for i in range(len(amino_acids)):
                aa_i = amino_acids[i]
                if aa_i == "-" or aa_i == "*":
                    continue
                ll_i = np.log(df.iloc[i,:][aa_i])
                per_position_ll.append(ll_i)
            
            pll_seq = np.average(per_position_ll)
            pll_all_sequences.append(pll_seq)

        return pll_all_sequences

