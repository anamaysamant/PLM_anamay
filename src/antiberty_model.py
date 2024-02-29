from antiberty import AntiBERTyRunner
import transformers
import antiberty
import torch
import pandas as pd
import pickle as pkl
import os
from antiberty import AntiBERTy
from antiberty.utils.general import exists
from tqdm import tqdm
class Antiberty():

    """
    Class for the protein Model Antiberty
    """

    def __init__(self, token = "average", file_name = "."):
        """
        Creates the instance of the language model instance; either light or heavy
        
        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
        trained_models_dir = os.path.join(project_path, 'trained_models')

        CHECKPOINT_PATH = os.path.join(trained_models_dir, 'AntiBERTy_md_smooth')
        VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = transformers.BertTokenizer(vocab_file=VOCAB_FILE,
                                                    do_lower_case=False)

        self.model = AntiBERTy.from_pretrained(CHECKPOINT_PATH).to(self.device)
        self.model.eval()

        self.token = token
        self.file = file_name

    def fit_transform(self, sequences, starts, ends):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """
        out_list = []
        print("Using the {} token\n".format(self.token))
        output = self.model.embed(sequences)
        if self.token == "average":
            for j,embedding in enumerate(output): #Taking the average for each of the aminoacid values
                out_list.append(torch.mean(embedding[starts[j]:ends[j]], axis = 0).tolist())
        elif self.token == "last":
            for j,embedding in enumerate(output): #Take the last token
                out_list.append(embedding[ends[j]-1].tolist())
        elif self.token == "first":
            for j,embedding in enumerate(output): #Take only CLS
                out_list.append(embedding[starts[j]].tolist())
        # pd.DataFrame(out_list).to_csv("outfiles/"+self.file+"/embeddings.csv")
        return pd.DataFrame(out_list,columns=[f"dim_{i}" for i in range(len(out_list[0]))])


    def pseudo_log_likelihood(self, sequences,starts = [], ends = [], batch_size=None):
        plls = []
        for s in tqdm(sequences):
            masked_sequences = []
            for i in range(len(s)):
                masked_sequence = list(s[:i]) + ["[MASK]"] + list(s[i + 1:])
                masked_sequences.append(" ".join(masked_sequence))

            # masked_sequences = [" ".join(s) for s in masked_sequences]
            tokenizer_out = self.tokenizer(
                masked_sequences,
                return_tensors="pt",
                padding=True,
            )
            tokens = tokenizer_out["input_ids"].to(self.device)
            attention_mask = tokenizer_out["attention_mask"].to(self.device)

            logits = []
            with torch.no_grad():
                if not exists(batch_size):
                    batch_size_ = len(masked_sequences)
                else:
                    batch_size_ = batch_size

                for i in range(0, len(masked_sequences), batch_size_):
                    batch_end = min(i + batch_size_, len(masked_sequences))
                    tokens_ = tokens[i:batch_end]
                    attention_mask_ = attention_mask[i:batch_end]
                    outputs = self.model(input_ids=tokens_,attention_mask=attention_mask_,)
                    logits.append(outputs.prediction_logits)

            logits = torch.cat(logits, dim=0)
            logits[:, :, self.tokenizer.all_special_ids] = -float("inf")
            logits = logits[:, 1:-1]  # remove CLS and SEP tokens

            # get masked token logits
            logits = torch.diagonal(logits, dim1=0, dim2=1).unsqueeze(0)
            labels = self.tokenizer.encode(
                " ".join(list(s)),
                return_tensors="pt",
            )[:, 1:-1]
            labels = labels.to("cuda:0" if torch.cuda.is_available() else "cpu")
            nll = torch.nn.functional.cross_entropy(
                logits,
                labels,
                reduction="mean",
            )
            pll = -nll
            plls.append(pll)

        plls = torch.stack(plls, dim=0)
        plls = plls.cpu().detach().tolist()

        return plls

    def calc_pseudo_likelihood_sequence(self, sequences: list):

        pll_all_sequences = self.pseudo_log_likelihood(sequences, batch_size=16)

        return pll_all_sequences

