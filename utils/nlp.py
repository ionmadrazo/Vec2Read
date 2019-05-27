"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging
import os
import sys
import re
try:
    import spacy
except:
    pass
import contextlib
import json
try:
    os.chdir("lib/RDRPOSTaggerP3/pSCRDRtagger/")
    sys.path.append(os.path.abspath(""))
    from RDRPOSTagger import RDRPOSTagger
    from Utility.Utils import getWordTag, getRawText, readDictionary
    os.chdir("../../../")
except:
    print("Library is not avaialble of in incorrect directory. This library is necessary if using POS tagging utilities in this package.")

class Tokenizer:
    def __init__(self,lang="en"):
        """
        :param config:
        """

        self.spacy_nlp = spacy.load('en')

    def tokenize(self, input):
        return [x.text for x in self.spacy_nlp.tokenizer(input) if x.text != " "]

    def split_sentences(self, input):
        return [x.text for x in self.spacy_nlp(input).sents if x.text != " "]

class POSTagger:
    def __init__(self,lang="en"):
        """
        :param config:
        """
        #langLong="Basque"

        self.short2LongLang= {"en":"English","es":"Spanish","eu":"Basque","it":"Italian","fr":"French","ca":"Catalan"}



        with contextlib.redirect_stdout(None):
            self.POSTagger = RDRPOSTagger()
            self.POSTagger.constructSCRDRtreeFromRDRfile("lib/RDRPOSTaggerP3/Models/UniPOS/UD_"+self.short2LongLang[lang]+"/train.UniPOS.RDR")
            self.POSModel = readDictionary("lib/RDRPOSTaggerP3/Models/UniPOS/UD_"+self.short2LongLang[lang]+"/train.UniPOS.DICT")


    def tagSentence(self, input):
        return self.POSTagger.tagRawSentence(self.POSModel, input)








def removeCitations(text=""):
    text= re.sub(r'\[\d+\]', '', text)
    return text

def loadDocsFromFolder(folder="",readAsJSON=False):
    allDocs = []
    for root, subFolders, files in os.walk(folder):
        for filename in files:
            text= open(root+"/"+filename, 'r', encoding="utf8").read()
            if readAsJSON:
                text = json.loads(text)
            allDocs = allDocs + [text]
    return allDocs

def stripSequence(text, prefix):
    result = text
    if result.startswith(prefix):
        result =  result[len(prefix):]
    if result.endswith(prefix):
        result =  result[:-len(prefix)]
    return result
