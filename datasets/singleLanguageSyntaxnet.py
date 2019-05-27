"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging
import os
import re
import utils.nlp as nlp
from torchtext.vocab import Vectors
from torchtext import data
from tqdm import tqdm
from torchtext.data import Iterator, BucketIterator
import random


class SingleLanguageSyntaxnetDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.logger.info("Initializing SingleLanguageDataLoader...")


        self.vectors = None
        if config.use_pretrained_embeddings and "embedding_max" in config:
            self.vectors = Vectors(name=config.embedding_file, cache=config.embedding_folder, max_vectors=config.embedding_max)
        elif config.use_pretrained_embeddings:
            self.vectors = Vectors(name=config.embedding_file, cache=config.embedding_folder)


        #Load all documents
        self.allDocs=[]
        for c in self.config.classes:
            folder = self.config.data_folders[c ]
            docs = nlp.loadDocsFromFolder(folder=folder,readAsJSON=True)
            self.logger.info("Found "+str(len(docs))+" files in "+folder)
            #attach label
            docs = [{"json":x,"label":c } for x in docs ]

            if "max_number_of_files_per_category" in config and config.max_number_of_files_per_category >0:
                self.allDocs=self.allDocs+ docs[:config.max_number_of_files_per_category]
            else:

                self.allDocs=self.allDocs+ docs

        if "max_number_of_files_per_category" in config and config.max_number_of_files_per_category >0:
            self.logger.info("Number of files per category limited by config: "+str(config.max_number_of_files_per_category))
            self.logger.info("Using "+str(len(self.allDocs))+" files in total.")

        allMorphLabels= set()

        for doc in self.allDocs:
            for sentence in doc["json"]:
                for word in sentence:
                    for feature in word["feats"]:
                        allMorphLabels.add(feature)
        allMorphLabels.remove("fPOS") #We do not want the POS tag in the morphology.
        morphString= generateMorphString(allMorphLabels=allMorphLabels,word=doc["json"][0][0])
        #print(morphString)




        for doc in tqdm(self.allDocs):

            wordSents=[]
            posSents=[]
            morphSents=[]
            for sentence in doc["json"]:

                wordList=[]
                posList=[]
                morphList=[]
                for word in sentence:
                    try: #Sometimes syntaxnet does not return any wordform which would make this loader crash otherwise
                        wordList.append(word["form"].lower())
                        posList.append(word["upostag"])
                        morphList.append(generateMorphString(allMorphLabels=allMorphLabels,word=word))
                    except:
                        pass
                wordSents.append("<_token_separator_>".join(wordList))
                posSents.append("<_token_separator_>".join(posList))
                morphSents.append("<_token_separator_>".join(morphList))
            doc["text_words"]="<_sentence_separator_>".join(wordSents)
            doc["text_POS"]="<_sentence_separator_>".join(posSents)
            doc["text_morph"]="<_sentence_separator_>".join(morphSents)
            #print(doc,doc["text_POS"].count("_token_separator_"),doc["text_words"].count("_token_separator_"),doc["text_morph"].count("_token_separator_"))




        self.SENTENCE = data.Field(tokenize=lambda x :  x.split("<_token_separator_>"), unk_token="<unk>", pad_token="<pad>", lower=True, batch_first=True)
        self.TEXT = data.NestedField(self.SENTENCE,  tokenize=lambda x : x.split("<_sentence_separator_>"), include_lengths=True   )

        self.SENTENCE_POS = data.Field(tokenize=lambda x :  x.split("<_token_separator_>"), unk_token="<unk>", pad_token="<pad>", lower=True, batch_first=True)
        self.TEXT_POS = data.NestedField(self.SENTENCE_POS,  tokenize=lambda x : x.split("<_sentence_separator_>"), include_lengths=True   )

        self.SENTENCE_MORPH = data.Field(tokenize=lambda x :  x.split("<_token_separator_>"), unk_token="<unk>", pad_token="<pad>", lower=True, batch_first=True)
        self.TEXT_MORPH = data.NestedField(self.SENTENCE_MORPH,  tokenize=lambda x : x.split("<_sentence_separator_>"), include_lengths=True   )

        self.LABEL = data.Field(sequential=False, use_vocab=True,batch_first=True)

        self.logger.info("Preprocessing...")
        fields = [("text", self.TEXT),("pos", self.TEXT_POS),("morph", self.TEXT_MORPH),("label", self.LABEL)]
        self.allExamples = []
        for doc in tqdm(self.allDocs):
            ex =  data.Example.fromlist([doc["text_words"],doc["text_POS"],doc["text_morph"],doc["label"]], fields)
            self.allExamples.append(ex)

        self.dataset = data.Dataset(self.allExamples, fields)

        self.TEXT.build_vocab(self.dataset, min_freq=1,vectors=self.vectors)
        self.TEXT_POS.build_vocab(self.dataset, min_freq=5)
        self.TEXT_MORPH.build_vocab(self.dataset, min_freq=1)
        self.LABEL.build_vocab(self.dataset, min_freq=1)
        self.config["paddingIdx"] = self.TEXT.vocab.stoi["<pad>"]
        self.config["vocab_size"] = len(self.TEXT.vocab)
        self.config["vocab_size_pos"] = len(self.TEXT_POS.vocab)
        self.config["vocab_size_morph"] = len(self.TEXT_MORPH.vocab)
        random.seed(config.random_seed)

        self.train_dataset,self.val_dataset= self.dataset.split(split_ratio=config.split_ratio,
                                            stratified=True,
                                            strata_field='label',
                                            random_state=random.getstate())

        self.train_iter,self.valid_iter= BucketIterator.splits(
         (self.train_dataset,self.val_dataset),
         batch_sizes=(config.batch_size,config.batch_size),
         sort_key=lambda x: len(x.text),
         sort_within_batch=True,
         repeat=False
         #,device=self.config.device
        )






        self.logger.info("SingleLanguageDataLoader initialized.")


    def finalize(self):
        pass

def generateMorphString(allMorphLabels={},word={}):
    morphDict = {}
    for key in allMorphLabels:
        morphDict[key]="undefinedMorphValue"
    for feature in word["feats"]:
        if feature != "fPOS":
            morphDict[feature]= word["feats"][feature]
    parts = [key+"_"+ morphDict[key] for key in morphDict]
    result = "<_morph_tag_separator_>".join(parts)
    return result
