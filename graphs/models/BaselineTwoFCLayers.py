
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F

class BaselineTwoFCLayers(nn.Module):

    def __init__(self, config, vectors=None, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger


        self.logger.info("Initializing BaselineTwoFCLayers model...")
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        if vectors is not None:
            self.embeddings.weight.data.copy_(vectors)
            self.embeddings.weight.requires_grad = config.finetune_embeddings

            #Will require next command for training if set so false
            #parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.linear1 = nn.Linear(config.embedding_dim, config.layer1_size)
        self.linear2 = nn.Linear(config.layer1_size, len(config.classes))
        #self.softmax = nn.LogSoftmax()

        self.logger.info("BaselineTwoFCLayers initialized.")
    def forward(self, inputs):
        words, nSents, nWords = inputs.text

        embeds = self.embeddings(words).mean(1).squeeze(1)
        out = F.relu(self.linear1(embeds))
        out = F.relu(self.linear2(out))


        #log_probs = F.log_softmax(result)
        #print(result/len(inputs), F.log_softmax(result/len(inputs)))
        #return self.softmax(result)
        #return F.log_softmax(out.mean(0)[0])
        #return out
        #print(out)
        #print(out.mean(1))
        #print(out.mean(2))
        #return out.mean(1)#.squeeze(0) #mean mantains the dimensions, that is why we need the extra [0]
        return torch.softmax(out.mean(1),1)

        #return F.sigmoid(result)
