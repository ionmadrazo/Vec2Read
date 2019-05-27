
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F

class EvenlyDistributedAttentionBlock(nn.Module):

    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger

        self.logger.info("Initializing EvenlyDistributedAttentionBlock...")


        self.logger.info("AttentionBlock initialized.")


    def forward(self, inputs,mask=None):




        size = list(inputs.size())
        size[-1]=1

        out = torch.ones(size,device=self.config.device)
        if mask is not None:
            out[mask!=1] = 0

        #Softmax is always computed in the second to last dimension of the input
        # as the last dimension is always an embedding representation (of a sentence or a word)
        return torch.softmax(out,len(inputs.size())-2)
