from esm import pretrained
import scipy.special
import torch
import pandas as pd
from tqdm import tqdm 
from esm import pretrained
import scipy.special
import torch
import pandas as pd
from tqdm import tqdm 
import pickle as pkl
import sys, os

sys.path.append(os.getcwd()+"/src")
from utils import get_pseudo_likelihood

class ESM_prob():

    """
    Class for the protein Language Model
    """

    def __init__(self, file_name):
        """
        Creates the instance of the language model instance, loads tokenizer and model
        """
        #Model of ESM used
        self.name_ = "esm1b_t33_650M_UR50S"
        #Which layer output to extract
        self.repr_layer_ = -1

        model, alphabet = pretrained.load_model_and_alphabet(self.name_)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        #model and alphabet
        self.model_ = model
        self.alphabet_ = alphabet
        
        #Name of output file
        self.file = file_name

    def fit_transform(self, sequences:list, batch_size = 10):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        
        batches: `int`
        Number of batches. Per batch a checkpoint file will be saved

        average: `bool`
        Take the average of the methods of the last hidden state for True or just the last method of the last hidden state for False.
        return
        ------

        None, saved the embeddings in the embeddings.csv
        """

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
            out = self.model_(batch_tokens,repr_layers = [self.repr_layer_],return_contacts = False)
            #Retrieve numerical values for each possible token (including aminoacids and special tokens) in each position
            logits = out["logits"][0].detach().numpy()
            #Turn them into probabilties 
            prob = scipy.special.softmax(logits,axis = 1)
            #Preprocessing probabilities, removing CLS and SEP tokens and removing probabilities of Special aminoacids and tokens of the model.
            df = pd.DataFrame(prob, columns = self.alphabet_.all_toks)
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