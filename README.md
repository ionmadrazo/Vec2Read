# Read2vec: Multilingual Readability Assessment

Read2vec is an automatic multilingual readability assessment tool developed in pytorch. This is the result of a research collaboration between [Ion Madrazo Azpiazu](https://ionmadrazo.github.io/) and [Maria Soledad Pera](https://solepera.github.io/).

Please cite this work as follows:

```
@article{madrazo2019,
    author = {Madrazo Azpiazu, Ion and Pera, Maria Soledad},
    title = {Multiattentive Recurrent Neural Network Architecture for Multilingual Readability Assessment},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {7},
    number = {},
    pages = {421-436},
    year = {2019},
    doi = {10.1162/tacl\_a\_00278},
    URL = {https://doi.org/10.1162/tacl_a_00278},
    eprint = {https://doi.org/10.1162/tacl_a_00278}
}

```
## Abstract
We present a hierarchical recurrent neural network architecture for automatic multilingual readability assessment. This architecture considers raw words as main input, but internally captures text structure and informs its word attention process using other syntax- and morphology-related datapoints, known to be of great importance to readability. This is achieved by a novel multiattentive strategy that allows the neural network to focus on specific parts of a text for predicting its readability level. We conducted exhaustive evaluation using datasets targeting multiple languages and prediction task types, to compare the proposed model with several baseline strategies.

![alt text](https://github.com/ionmadrazo/Read2Vec-pytorch/blob/master/architecture.png)


## Requirements
In order to run this code you will first need to:

- [Python 3.7](https://www.python.org/downloads/)
- [Anaconda 4.5.11](https://www.anaconda.com/download/)
- [Pytorch 1.0.0](https://pytorch.org/)


In addition, you will also need to install libraries described in requirements.txt file located on the root directory of this repository. You can install the by running the following command:
```
pip install -r requirements.txt
```

## Training your own model
In order to train a model using Read2vec you need to follow the steps below:

### Decide on a data loader
You can find different data loaders under **datasets** directory. The recommeded one is SingleLanguageSyntaxnetDataLoader which loads a dataset parsed by syntaxnet. You can find an example of how to use syntaxnet to parse documents in **notebooks/ParseDocumentsWithSyntaxnet.ipynb**. The data loader expects each text to be in a separate file and in json format. In addition, all documents for an specific class need to be in a single directory.


```python
# Example of a file structure for text: The drink is served hot.
# Note that each document is a list of sentences and each sentence is a list of word dictionaries.
[
[
        {
            "id": 1,
            "form": "The",
            "upostag": "DET",
            "xpostag": "DT",
            "feats": {
                "Definite": "Def",
                "fPOS": "DET++DT",
                "PronType": "Art"
            },
            "head": 2,
            "deprel": "det"
        },
        {
            "id": 2,
            "form": "drink",
            "upostag": "NOUN",
            "xpostag": "NN",
            "feats": {
                "fPOS": "NOUN++NN",
                "Number": "Sing"
            },
            "head": 4,
            "deprel": "nsubjpass"
        },
        {
            "id": 3,
            "form": "is",
            "upostag": "AUX",
            "xpostag": "VBZ",
            "feats": {
                "Mood": "Ind",
                "fPOS": "VERB++VBZ",
                "Number": "Sing",
                "Person": "3",
                "Tense": "Pres",
                "VerbForm": "Fin"
            },
            "head": 4,
            "deprel": "auxpass"
        },
        {
            "id": 4,
            "form": "served",
            "upostag": "VERB",
            "xpostag": "VBN",
            "feats": {
                "fPOS": "ADJ++JJ",
                "Degree": "Pos"
            },
            "head": 0,
            "deprel": "ROOT"
        },
        {
            "id": 5,
            "form": "hot",
            "upostag": "NOUN",
            "xpostag": "NN",
            "feats": {
                "fPOS": "NOUN++NN",
                "Number": "Sing"
            },
            "head": 4,
            "deprel": "xcomp"
        },
        {
            "id": 6,
            "form": ".",
            "upostag": "PUNCT",
            "xpostag": ".",
            "feats": {
                "fPOS": "PUNCT++."
            },
            "head": 4,
            "deprel": "punct"
        }
    ]
]
```

### Configure training process
You will need to create an experiment configuration file. You can simply start by extending any of the files under **config/** directory. Parts you will most likely need to modify are the following:

```python
"classes": ["classname1","classname2"],
"data_folders": {"classname1":"folder/where/you/stored/all/files/for/classname1",
                  "classname2":"folder/where/you/stored/all/files/for/classname2"},
"embedding_folder":"embedding/folder",
"embedding_file":"name_of_embeddingfile.vec",
"max_number_of_files_per_category": <your number of files per category>,
```

### Train
To run the experiment use the following command:

```
python main.py configs/config_name.json
```

## License

This software has a free to use license for non commercial use. Check **LICENSE** file for more details.
