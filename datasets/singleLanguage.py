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


class SingleLanguageDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.logger.info("Initializing SingleLanguageDataLoader...")

        self.vectors = None
        if config.use_pretrained_embeddings:
            self.vectors = Vectors(name=config.embedding_file, cache=config.embedding_folder)


        #Load all documents
        self.allDocs=[]
        for c in self.config.classes:
            folder = self.config.data_folders[c ]
            docs = nlp.loadDocsFromFolder(folder=folder)
            self.logger.info("Found "+str(len(docs))+" files in "+folder)
            #attach label
            docs = [{"text":x,"label":c } for x in docs ]

            if "max_number_of_files_per_category" in config and config.max_number_of_files_per_category >0:
                self.allDocs=self.allDocs+ docs[:config.max_number_of_files_per_category]
            else:

                self.allDocs=self.allDocs+ docs

        if "max_number_of_files_per_category" in config and config.max_number_of_files_per_category >0:
            self.logger.info("Number of files per category limited by config: "+str(config.max_number_of_files_per_category))
            self.logger.info("Using "+str(len(self.allDocs))+" files in total.")

        #Clean documents
        for doc in self.allDocs:
            doc["text"]= nlp.removeCitations(doc["text"])
            doc["text"]= doc["text"].replace("\n","")

        tokenizer = nlp.Tokenizer(config.lang)
        POStagger = nlp.POSTagger(config.lang)

        #Generate POS tags
        for doc in self.allDocs:
            alignedText = []
            alignedPOS = []
            #print(doc)
            for sent in tokenizer.split_sentences(doc["text"]):
                tokenizedSent = tokenizer.tokenize(sent)
                taggedSent = POStagger.tagSentence(" ".join(tokenizedSent))


                alignedSent = [x.split("/")[0] for x in taggedSent.split()]
                alignedSentPOS = [x.split("/")[1]+"_pos" for x in taggedSent.split()]

                alignedSent.append("<_sentence_separator_>")
                alignedSentPOS.append("<_sentence_separator_>")
                alignedText=alignedText+alignedSent
                alignedPOS=alignedPOS+alignedSentPOS

                assert len(alignedSent) == len(alignedSentPOS)


            if len(alignedText) > 1:
                #remove last sentence separator
                alignedText=alignedText[:-1]
                alignedPOS=alignedPOS[:-1]

            assert len(alignedText) == len(alignedPOS)
            doc["text"]= "<_token_separator_>".join(alignedText)
            doc["text_POS"]= "<_token_separator_>".join(alignedPOS)


#9,34,27
#24 59-->60

        self.SENTENCE = data.Field(tokenize=lambda x :  nlp.stripSequence(x, "<_token_separator_>").split("<_token_separator_>"), unk_token="<unk>", pad_token="<pad>", lower=True, batch_first=True)
        self.TEXT = data.NestedField(self.SENTENCE,  tokenize=lambda x : x.split("<_sentence_separator_>"), include_lengths=True   )

        self.SENTENCE_POS = data.Field(tokenize=lambda x :  nlp.stripSequence(x, "<_token_separator_>").split("<_token_separator_>"), unk_token="<unk>", pad_token="<pad>", lower=True, batch_first=True)
        self.TEXT_POS = data.NestedField(self.SENTENCE_POS,  tokenize=lambda x : x.split("<_sentence_separator_>"), include_lengths=True   )

        self.LABEL = data.Field(sequential=False, use_vocab=True,batch_first=True)

        self.logger.info("Preprocessing...")
        fields = [("text", self.TEXT),("text_POS", self.TEXT_POS),("label", self.LABEL)]
        self.allExamples = []
        for doc in tqdm(self.allDocs):
            ex =  data.Example.fromlist([doc["text"],doc["text_POS"],doc["label"]], fields)
            self.allExamples.append(ex)

        self.dataset = data.Dataset(self.allExamples, fields)

        self.TEXT.build_vocab(self.dataset, min_freq=1,vectors=self.vectors)
        self.TEXT_POS.build_vocab(self.dataset, min_freq=5)
        self.LABEL.build_vocab(self.dataset, min_freq=1)
        self.config["paddingIdx"] = self.TEXT.vocab.stoi["<pad>"]
        self.config["vocab_size"] = len(self.TEXT.vocab)
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
        )






        self.logger.info("SingleLanguageDataLoader initialized.")


    def finalize(self):
        pass
