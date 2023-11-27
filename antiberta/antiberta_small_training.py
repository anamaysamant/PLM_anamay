# This is the antiberta notebook from the publication as a python script
import sys
data_folder = sys.argv[1]

# Imports
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import datasets
datasets.disable_progress_bar() 
import os

# Initialise the tokeniser
tokenizer = RobertaTokenizer.from_pretrained(
    "antibody-tokenizer"
)

# Initialise the data collator, which is necessary for batching
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Load the dataset
text_datasets = {
    "train": [f'{data_folder}/train.txt'],
    "eval": [f'{data_folder}/test.txt'],
    "test": [f'{data_folder}/val.txt']
}

dataset = load_dataset("text", data_files=text_datasets, cache_dir="cluster/scratch/wglaenzer/huggingface_cache") #change this to your own cache directory
tokenized_dataset = dataset.map(
    lambda z: tokenizer(
        z["text"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    batched=True,
    num_proc=1,
    remove_columns=["text"],
)

# These are the cofigurations used for pre-training
antiberta_config = {
    "num_hidden_layers": 12, #originally 12
    "num_attention_heads": 12,
    "hidden_size": 768, #originally 768
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96,
    "max_steps": 31250, #originally 225000
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}

# Initialise the model
model_config = RobertaConfig(
    vocab_size=antiberta_config.get("vocab_size"),
    hidden_size=antiberta_config.get("hidden_size"),
    max_position_embeddings=antiberta_config.get("max_position_embeddings"),
    num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
    num_attention_heads=antiberta_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLM(model_config)

# Construct training arguments
# Huggingface uses a default seed of 42
args = TrainingArguments(
    #disable_tqdm=True, #added to disable the progress bar
    output_dir="antiberta_large_checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=antiberta_config.get("batch_size", 32),
    per_device_eval_batch_size=antiberta_config.get("batch_size", 32),
    max_steps=234375, #originally 225000
    save_steps=2500,#originally 2500
    logging_steps=2500, #originally 2500
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    learning_rate=1e-4,
    gradient_accumulation_steps=antiberta_config.get("gradient_accumulation_steps", 1),
    fp16=True, #comment this out for training on CPU
    evaluation_strategy="steps",
    seed=42
)

### Setup the HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"]
)

trainer.train()
trainer.save_model(f'antiberta_large')
# Predict MLM performance on the test datase
# (everything below here makes a loss plot, it doen't work for larger models due to memory issues)
#out = trainer.predict(tokenized_dataset['test'])

import pandas as pd
#make a lineplot of the loss
#history = pd.DataFrame(trainer.state.log_history)
#print(history)
print(trainer.state.log_history)

#write trainer.state.log_history to a csv file
history=trainer.state.log_history

with open('loss.txt', 'w') as f:
     f.write(str(history))

losstime = []
evallosstime = []
i = 0
for timepoint in history:
    print(timepoint)
    if i % 2 == 0:
        losstime.append([timepoint['loss'],timepoint['step']])
    else:
        evallosstime.append([timepoint['eval_loss'],timepoint['step']])
    i = i + 1
    if timepoint['step'] == antiberta_config["max_steps"]:
        break
import matplotlib.pyplot as plt
import numpy as np

losstime = np.array(losstime)
fig, ax = plt.subplots()
ax.step(losstime[:,1], losstime[:,0], linewidth=1.5)
ax.set_ylabel('Loss')
ax.set_xlabel('Steps')
ax.set_title('Loss')
plt.show()
fig.savefig('loss_steps.png')

evallosstime = np.array(evallosstime)
fig, ax = plt.subplots()
ax.step(evallosstime[:,1], evallosstime[:,0], linewidth=1.5)
ax.set_ylabel('Evaluation loss')
ax.set_xlabel('Steps')
ax.set_title('Evaluation loss')
plt.show()
fig.savefig('eval_loss_steps.png')