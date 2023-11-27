#this was made with the code from the paratope prediciton notebook modified to fit for sequence classification
import sys

from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import (
    Dataset,
    DatasetDict,
    Sequence,
    ClassLabel
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import pandas as pd
import torch
import numpy as np
import random
import os

TOKENIZER_DIR = "antibody-tokenizer"

# Initialise a tokenizer
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_DIR, max_len=150)

# Read in the json files
train_df = pd.read_json(
    "data/antigen_binding/mHER_H3.proc.adj.train.jsonl", 
    lines=True)
val_df = pd.read_json(
    "data/antigen_binding/mHER_H3.proc.adj.valid.jsonl",
    lines=True)
test_df = pd.read_json(
    "data/antigen_binding/mHER_H3.proc.adj.test.jsonl",
    lines=True)

# Get a preview
train_df.head(3)

print(train_df.shape)

# Create a new Dataset Dict with the sequence and paratope labels
ab_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[['sequence','label']]),
    "validation": Dataset.from_pandas(val_df[['sequence','label']]),
    "test": Dataset.from_pandas(test_df[['sequence','label']])
})

# This is what a DatasetDict object looks like with its individual Dataset things
print(ab_dataset)

print(ab_dataset['train'].select(range(1))['sequence'])

print(ab_dataset['train'].select(range(1))['label'][0])

# Look at the Features of each column in the train dataset within the ab_dataset DatasetDict
print(ab_dataset['train'].features)

# We iterate through the sequence and labels columns
# Keeping the sequence column as-is, but applying a str2int function, allowing us to cast later
ab_dataset_featurised = ab_dataset.map(
    lambda seq, label: {
        "sequence": seq,
        "label": label
    }, 
    input_columns=["sequence", "label"], batched=True
)

# Get the old Features instance from the previous ab_dataset
# Notice how labels is a Sequence of Value
feature_set_copy = ab_dataset['train'].features.copy()
feature_set_copy

# Cast to the `new_feature` that we made earlier
#feature_set_copy['paratope_labels'] = new_feature

ab_dataset_featurised = ab_dataset_featurised.cast(feature_set_copy)

ab_dataset_featurised['train'].features 

# now the labels are actually a series of integers, but is recognised by huggingface as a series of Classlabels
print(ab_dataset_featurised['train'].select(range(1))['label'][0])

### Tokenizing the inputs
# we need to redefine this to e.g. put -100 labels for the start/end tokens

def preprocess(batch):
    t_inputs = tokenizer(batch['sequence'], 
        padding="max_length")
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    batch['labels'] = batch["label"]
    return batch


# Apply that function above on the dataset
ab_dataset_tokenized = ab_dataset_featurised.map(
    preprocess, 
    batched=True,
    batch_size=32,
    remove_columns=['sequence', 'label']
)

ab_dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print(ab_dataset_tokenized.shape)

## Model training
label_list = [0,1]

def compute_metrics(p):
    """
    A callback added to the trainer so that we calculate various metrics via sklearn
    """
    predictions, labels = p
    
    prediction_pr = torch.softmax(torch.from_numpy(predictions), dim=1).detach().numpy()[:,-1]
    
    # We run an argmax to get the label
    predictions = np.argmax(predictions, axis=1)

    return {
        "precision": precision_score(labels, predictions), 
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "auc": roc_auc_score(labels, prediction_pr),
        "aupr": average_precision_score(labels, prediction_pr),
        "mcc": matthews_corrcoef(labels, predictions),
    }

# define batch size, metric you want etc. 
batch_size = 32
RUN_ID = "antigen-binding-task"
SEED = 0
LR = 1e-6

args = TrainingArguments(
    f"{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=LR, # 1e-6, 5e-6, 1e-5. .... 1e-3
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2, #originally 10
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='aupr', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

def set_seed(seed: int = 42):
    """
    Set all seeds to make results reproducible (deterministic mode).
    When seed is None, disables deterministic mode.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)

# Name of the pre-trained model after you train your MLM
MODEL_DIR = "trained_model_30_milliondata/30million_mixed_model"

# We initialise a model using the weights from the pre-trained model
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

trainer = Trainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=ab_dataset_tokenized['train'],
    eval_dataset=ab_dataset_tokenized['validation'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
trainer.save_model(f'trained_model_30_milliondata/antigen_binding_finetuned')


# run prediction on the test set
pred = trainer.predict(
    ab_dataset_tokenized['test']
)

# this stores a JSON with metric values
pred.metrics 

# input sequence of tralokinumab Light chain
#input_seq = 'YVLTQPPSVSVAPGKTARITCGGNIIGSKLVHWYQQKPGQAPVLVIYDDGDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDTGSDPVVFGGGTKLTVL'
#model = RobertaForTokenClassification.from_pretrained(
#    f'trained_model_30_milliondata/antigen_binging_finetuned'
#)

#tokenized_input = tokenizer([input_seq], return_tensors='pt', padding=True)
#predicted_logits = model(**tokenized_input)

# Simple argmax - no thresholding.
#argmax = predicted_logits[0].argmax(2)[0][1:-1].cpu().numpy()
#indices = np.argwhere(argmax).flatten()

#predicted_sequence = ''

#for i, s in enumerate(input_seq):
#    if i in indices:
#        predicted_sequence += s
#    else:
#        predicted_sequence += '-'

