# PlatypusPython

## Workflow

The workflow takes as input either a Platypus VGM or a list of sequences in a CSV. Then the sequences are passed into a pretrained language model to make embeddings and an optional UMAP for visualization.
#
## Running the workflow

- Create a new environment using Python3.8 as:
`conda create -n plm python=3.8`

- Activate the environment:
`conda activate plm`

- To install necessary libraries:
`pip install -r requirements.txt`

- Fill in the configuration file with necessary info

- Run the workflow with `snakemake -c (available cores or "all")`

#
### Modes

There are 3 different main modes:
- `VGM` which assumes that a VGM exists in the data folder named data.R which will create the CSV with the sequences.
- `CSV` which assumes that the CSV is ready with the sequences, but the user has to provide the location of the CSV and can provide the name of the column
#

## Arguments

- `data_path` path with the location of the CSV file or the location of the VGM depending on the mode
- `vgm_colname` column in the vgm that contains the sequences (optional)
- `folder_name` if TRUE then each run output will be stored in a folder with the time and date of execution
- `umap` if True, it also created a UMAP based on the made embeddings.

## Arguments per method

### Ablang [\[code\]](https://github.com/oxpig/AbLang/tree/main) 
- `chain`: input chain `heavy`(default) or `light`
- `method`: `seqcoding` returns the average embedding from all tokens

### Antiberty [\[code\]](https://pypi.org/project/antiberty/)
- `method`: strategy with which we get the embeddings. Options `last`, `first`, `average`(default)

### Protbert[\[code\]](https://huggingface.co/Rostlab/prot_bert)
- `method`: strategy with which we get the embeddings. Options `last`, `first`, `average`(default), `pooler`

### ESM2[\[code\]](https://huggingface.co/docs/transformers/model_doc/esm)
- `method`: strategy with which we get the embeddings. Options `last`, `first`, `average`(default), `prob`

### Sapiens[\[code\]](https://pypi.org/project/sapiens/)
- `method`: layer from which to extract the embeddings per sequence .
- `chain`: input chain `H`(default) or `L`

### TCRBert[\[code\]](https://huggingface.co/wukevin/tcr-bert)
- `method`: layer from which to extract the embeddings per sequence.

### How to pass arguments?
In the config.yml, add the line: `arguement` : `value`.\
For example in `Sapiens` the arguments can look like: \
`method`: `average`\
`chain`: `H`\
If you do not provide any argument the default will be passed.
#

## Adding new models

- Create a .py file which will contain the model class in the folder `src/`. Be careful that the name of the file is not the same as any of the packages that we are using
- Make a model class file like the others (simple example is the ablang_model). 
- Each class consists of a init function, where you initialize things, like first making the wanted model and adding it to the self.model variable. 
- Then it contains a fit_predict which will use the predict function of the model, to get the embeddings and print the output in a csv file.
- Then add the import to the make_membeddings.py which will use the model.
- The user then can pick which model they want through the config.yml `model` argument

## Computing likelihoods
- Use the `ESM` or the `Sapiens` with the `prob` method option, to get the probabilities_pseudo.pkl containing a list with the per sequence per position aminoacid likelihoods and the whole sequence log-pseudolikelihood.

## Classification of embeddings

ML classification for binary and multi-class labels.\
Builds Neural Network, Support Vector Machine, Random Forest, Gaussian Naive Bayes, and Logistic Regression classifier models.\
When `display = 'both'` creates confusion matrices and ROC curves for each model.
- Two required inputs, embeddings and labels, as pd.dataframe and np.array respectively.  

### Arguments

- `eval_size`: Used to determine subsect of data to be used to Evaluate model’s performances. Default is `0.1`, input must be between 0-1.  
- `balancing`: Balancing uses sklearn’s RandomOverSampling() of all minority classes so that the number of minority equals majority class. Takes a boolean as input, default is `True`. 
- `scaling`: Scaling uses the sklearn’s StandardScaler function to scale data according to label. Takes boolean input, default is `False`.
- `display`: Makes confusion matrices and ROC curves for all models. Takes four inputs (`'none'`, `'both'`, `'cm'`, and `'roc'`), default is `'none'`. 
- `epochs`: Determines how many passes the model will undergo during configuration, used to update weight, bias, and parameter optimization. Takes a numeric input, default is 3 times the number of embedding columns.
- `nodes`: Influences how many nodes will consist in the single layered neural network. Takes a numeric input, default is 1/2 the number of embedding columns.
- `batch_size`: determines how many training samples are processed together. Takes a numeric value, default is `32`. 
- `patience`: Determines how many epochs the model will train for with no decreases in loss. Takes integer input greater than 0, default is `15`.
- `noise`: Creates an initial Gaussian noise layer before hidden layers of neural network. Takes boolean input, default is `False`.
- `return_model`: Returns the neural network made from the get_Classification function. Takes boolean input, default is `False`.
#

### Examples

`import Classification.py as clf`\
`embeddings = pd.read_csv('embeddings.csv')`\
`metadata = pd.read_csv('metadata')`\
`labels = np.array(metadata['labels'])`\
Only required arguments: `clf.get_Classification(embeddings, labels)`\
Optional arguments: `clf.get_Classification(embeddings, labels, balancing = True, scaling = True, display = 'both', eval_size = 0.2, epochs = 200, nodes = 100, batch_size = 16, patience = 20)`
