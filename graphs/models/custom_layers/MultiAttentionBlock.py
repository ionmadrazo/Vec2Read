
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F

class MultiAttentionBlock(nn.Module):

    def __init__(self, config, attentionBlocks={}, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.attentionBlocks=attentionBlocks

        self.logger.info("Initializing MultiAttentionBlock with " + str(len(self.attentionBlocks))+" blocks...")
        nAttBlocks = len(self.attentionBlocks)
        att_init_weight = [1/nAttBlocks for x in self.attentionBlocks] # evenly distributed initialization
        self.attention_weights = torch.nn.Parameter(torch.Tensor(att_init_weight),requires_grad=True)

        self.logger.info("MultiAttentionBlock initialized.")


    def forward(self, inputs={},mask=None):

        normAttW=torch.softmax(self.attention_weights,0)

        result = None
        for idx,key in enumerate(self.attentionBlocks):
            if result is None:
                result = self.attentionBlocks[key](inputs[key],mask) * normAttW[idx]
            else:
                result = result+ self.attentionBlocks[key](inputs[key],mask) * normAttW[idx]
        
        return result
