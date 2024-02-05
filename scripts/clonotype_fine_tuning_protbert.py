import os 
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true=labels, y_pred=predictions)}


parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset')
# parser.add_argument('--group', default="v_gene_family")
# parser.add_argument('--color') 
# parser.add_argument('--include_germline', action='store_true')           
# parser.add_argument('--mode') 

args = parser.parse_args()

dataset = args.dataset
# group = args.group
# include_germline = args.include_germline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

CACHE_DIR = "/cluster/scratch/asamant/models--Rostlab--prot_bert/snapshots/7a894481acdc12202f0a415dd567f6cfdb698908"

mode = "full_VDJ"

model_names = ["ablang","sapiens","protbert","ESM"]
data = pd.read_csv(os.path.join("..","..","..","data",dataset,"vdj_evolike_combine.csv"))

data = data.loc[data["sample_id"] == "s1",:]
data = data.drop_duplicates(["full_VDJ_aa"]).reset_index(drop=True)
data["sample_clonotype"] = list(data.apply(lambda x: x["sample_id"] + "_" + x["raw_clonotype_id"], axis=1))

clonotype_seq_counts = data["sample_clonotype"].value_counts()
more_than_one_clonotypes = list(clonotype_seq_counts[clonotype_seq_counts > 1].index)
more_than_one_indicator = data["sample_clonotype"].apply(lambda x: x in more_than_one_clonotypes)
data = data.loc[more_than_one_indicator,:].reset_index(drop=True)

train_seqs = data["full_VDJ_aa"].apply(lambda x: ' '.join(list(x)))
num_labels = len(data["sample_clonotype"].unique())

train_labels = pd.factorize(data["sample_clonotype"].copy())[0]

train_seqs, val_seqs, train_labels, val_labels = train_test_split(train_seqs, train_labels, test_size=0.1)

tokenizer = BertTokenizer.from_pretrained(CACHE_DIR, do_lower_case=False)

tokenized_train_sequences = tokenizer(list(train_seqs), padding = True, return_tensors= 'pt') 
tokenized_val_sequences = tokenizer(list(val_seqs), padding = True, return_tensors= 'pt') 

train_dataset = SeqDataset(tokenized_train_sequences, train_labels)
val_dataset = SeqDataset(tokenized_val_sequences, val_labels)

model = BertForSequenceClassification.from_pretrained(CACHE_DIR, num_labels = num_labels)

training_args = TrainingArguments(
    output_dir='/cluster/scratch/asamant/protbert_fine_tuning', 
    save_total_limit = 2,
    resume_from_checkpoint=True,         
    num_train_epochs=150,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    # logging_dir='./logs',           
    # logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics          
)

trainer.train()

