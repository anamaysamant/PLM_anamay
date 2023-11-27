import torch
import pandas as pd
import numpy as np
import skorch
from transformers import BertModel, BertTokenizer


class TCRBert():
    """
    Class for the TCRBert Language Model
    """

    def __init__(self, method='average', file_name = 'TCRBert'):
        """
        Creates the instance of the language model instance, loads tokenizer and model
        """
        self.model = BertModel.from_pretrained("wukevin/tcr-bert")
        self.tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")
        self.method = method
        self.file = file_name
        
    def fit_transform(self, sequences:list):
        """
        Fits the model and outputs the embeddings.
        parameters
        ----------
        sequences: `list` 
        List with sequences to be transformed
        ------
        None, saved the embeddings in the embeddings.csv
        """
        embeddings = []
        print("Using '"+self.method+"' Method")
        for sequence in sequences:
            sequence = ' '.join(sequence)
            token = self.tokenizer(sequence, return_tensors="pt")
            output = self.model(**token)
            if self.method == "average":
                output = torch.mean(output.last_hidden_state, axis = 1)[0]
                    
            elif self.method == "pooler":
                output = output.pooler_output[0]
                
            elif self.method == "first":
                output = output.last_hidden_state[0,0,:]

            elif self.method == "last":
                output = output.last_hidden_state[0,-1,:]
            
            embeddings.append(output.tolist())
        
        pd.DataFrame(embeddings).to_csv("outfiles/"+self.file+"/embeddings.csv")
