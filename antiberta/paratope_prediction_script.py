#this was made with the code from the paratope prediciton notebook
import sys

from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
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

# Read in parquet files
train_df = pd.read_parquet(
    'assets/sabdab_train.parquet'
)
val_df = pd.read_parquet(
    'assets/sabdab_val.parquet'
)
test_df = pd.read_parquet(
    'assets/sabdab_test.parquet'
)

# Get a preview
train_df.head(3)

# Create a new Dataset Dict with the sequence and paratope labels
ab_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[['sequence','paratope_labels']]),
    "validation": Dataset.from_pandas(val_df[['sequence','paratope_labels']]),
    "test": Dataset.from_pandas(test_df[['sequence','paratope_labels']])
})

# This is what a DatasetDict object looks like with its individual Dataset things
print(ab_dataset)

print(ab_dataset['train'].select(range(1))['sequence'])

print(ab_dataset['train'].select(range(1))['paratope_labels'][0][20:35])

# Look at the Features of each column in the train dataset within the ab_dataset DatasetDict
print(ab_dataset['train'].features)

# Create a ClassLabel feature which will replace paratope_labels later.
paratope_class_label = ClassLabel(2, names=['N','P'])
new_feature = Sequence(
    paratope_class_label
)

# We iterate through the sequence and labels columns
# Keeping the sequence column as-is, but applying a str2int function, allowing us to cast later
ab_dataset_featurised = ab_dataset.map(
    lambda seq, labels: {
        "sequence": seq,
        "paratope_labels": [paratope_class_label.str2int(sample) for sample in labels]
    }, 
    input_columns=["sequence", "paratope_labels"], batched=True
)

# Get the old Features instance from the previous ab_dataset
# Notice how labels is a Sequence of Value
feature_set_copy = ab_dataset['train'].features.copy()
feature_set_copy

# Cast to the `new_feature` that we made earlier
feature_set_copy['paratope_labels'] = new_feature

if '__index_level_0__' in feature_set_copy:
    del feature_set_copy['__index_level_0__']

ab_dataset_featurised = ab_dataset_featurised.cast(feature_set_copy)

ab_dataset_featurised['train'].features 

# now the labels are actually a series of integers, but is recognised by huggingface as a series of Classlabels
print(ab_dataset_featurised['train'].select(range(1))['paratope_labels'][0][20:35])

### Tokenizing the inputs
# we need to redefine this to e.g. put -100 labels for the start/end tokens

def preprocess(batch):
    # :facepalm: The preprocess function takes tokenizer and needs a LIST not a PT tensor :eyeroll:
    # https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    
    t_inputs = tokenizer(batch['sequence'], 
        padding="max_length")
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    
    # enumerate 
    labels_container = []
    for index, labels in enumerate(batch['paratope_labels']):
        
        # This is typically length of the sequence + SOS + EOS + PAD (to longest example in batch)
        tokenized_input_length = len(batch['input_ids'][index])
        paratope_label_length  = len(batch['paratope_labels'][index])
        
        # we subtract 1 because we start with SOS
        # we should in theory have at least 1 "pad_with_eos" because an EOS wouldn't have been accounted for in the
        # paratope_labels column even for the longest possible sequence
        n_pads_with_eos = max(1, tokenized_input_length - paratope_label_length - 1)
        
        # We have a starting -100 for the SOS
        # and fill the rest of seq length with -100 to account for any extra pads and the final EOS token
        # The -100s are ignored in the CE loss function
        labels_padded = [-100] + labels + [-100] * n_pads_with_eos
        
        assert len(labels_padded) == len(batch['input_ids'][index]), \
        f"Lengths don't align, {len(labels_padded)}, {len(batch['input_ids'][index])}, {len(labels)}"
        
        labels_container.append(labels_padded)
    
    # We create a new column called `labels`, which is recognised by the HF trainer object
    batch['labels'] = labels_container
    
    for i,v in enumerate(batch['labels']):
        assert len(batch['input_ids'][i]) == len(v) == len(batch['attention_mask'][i])

    print(batch)
    
    return batch

# Apply that function above on the dataset - labels now aligned!
ab_dataset_tokenized = ab_dataset_featurised.map(
    preprocess, 
    batched=True,
    batch_size=8,
    remove_columns=['sequence', 'paratope_labels']
)

## Model training
# This has the actual names that maps 0->N and 1->P
label_list = paratope_class_label.names

def compute_metrics(p):
    """
    A callback added to the trainer so that we calculate various metrics via sklearn
    """
    predictions, labels = p
    
    # The predictions are logits, so we apply softmax to get the probabilities. We only need
    # the probabilities of the paratope label, which is index 1 (according to the ClassLabel we made earlier),
    # or the last column from the output tensor
    prediction_pr = torch.softmax(torch.from_numpy(predictions), dim=2).detach().numpy()[:,:,-1]
    
    # We run an argmax to get the label
    predictions = np.argmax(predictions, axis=2)

    # Only compute on positions that are not labelled -100
    preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    labs = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    probs = [ 
        [prediction_pr[i][pos] for (pr, (pos, l)) in zip(prediction, enumerate(label)) if l!=-100]
         for i, (prediction, label) in enumerate(zip(predictions, labels)) 
    ] 
            
    # flatten
    preds = sum(preds, [])
    labs = sum(labs, [])
    probs = sum(probs,[])
    
    return {
        "precision": precision_score(labs, preds, pos_label="P"),
        "recall": recall_score(labs, preds, pos_label="P"),
        "f1": f1_score(labs, preds, pos_label="P"),
        "auc": roc_auc_score(labs, probs),
        "aupr": average_precision_score(labs, probs, pos_label="P"),
        "mcc": matthews_corrcoef(labs, preds),
    }

# define batch size, metric you want etc. 
batch_size = 32
RUN_ID = "paratope-prediction-task"
SEED = 0
LR = 1e-6

args = TrainingArguments(
    f"{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=LR, # 1e-6, 5e-6, 1e-5. .... 1e-3
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10, #originally 10
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
model = RobertaForTokenClassification.from_pretrained(MODEL_DIR, num_labels=2)

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
trainer.save_model(f'trained_model_30_milliondata/paratope_prediction_finetuned')


# run prediction on the test set
pred = trainer.predict(
    ab_dataset_tokenized['test']
)

# this stores a JSON with metric values
pred.metrics 

# input sequence of tralokinumab Light chain
input_seq = 'YVLTQPPSVSVAPGKTARITCGGNIIGSKLVHWYQQKPGQAPVLVIYDDGDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDTGSDPVVFGGGTKLTVL'
model = RobertaForTokenClassification.from_pretrained(
    f'trained_model_30_milliondata/paratope_prediction_finetuned'
)

tokenized_input = tokenizer([input_seq], return_tensors='pt', padding=True)
predicted_logits = model(**tokenized_input)

# Simple argmax - no thresholding.
argmax = predicted_logits[0].argmax(2)[0][1:-1].cpu().numpy()
indices = np.argwhere(argmax).flatten()

predicted_sequence = ''

for i, s in enumerate(input_seq):
    if i in indices:
        predicted_sequence += s
    else:
        predicted_sequence += '-'

