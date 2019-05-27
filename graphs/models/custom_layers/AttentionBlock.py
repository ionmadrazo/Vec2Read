
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F

class AttentionBlock(nn.Module):

    def __init__(self, config, attention_inputSize, attention_hiddenSize, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.attention_inputSize = attention_inputSize
        self.attention_hiddenSize = attention_hiddenSize

        self.logger.info("Initializing AttentionBlock with input of " + str(self.attention_inputSize)+"...")

        self.attentionLayer1= nn.Linear(self.attention_inputSize, self.attention_hiddenSize)
        self.attentionLayer2= nn.Linear(self.attention_hiddenSize, 1)

        self.logger.info("AttentionBlock initialized.")


    def forward(self, inputs,mask=None):


        h = F.relu(self.attentionLayer1(inputs))
        out = F.relu(self.attentionLayer2(h))
        #The mask is usefull for ignoring attention generated for padded parts of sequence
        if mask is not None:
            out[mask!=1] = 0

        #Softmax is always computed in the second to last dimension of the input
        # as the last dimension is always an embedding representation (of a sentence or a word)
        return torch.softmax(out,len(inputs.size())-2)
