
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from graphs.models.custom_layers.AttentionBlock import AttentionBlock
class Read2VecWordAttention(nn.Module):

    def __init__(self, config, vectors=None, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger


        self.logger.info("Initializing Read2VecWordAttentionOnly model...")

        if "paddingIdx" not in self.config:
            self.logger.error("Padding index needs to be configured in order to this module to work. You can set it via config file or by using a DataLoader that automatically sets that for you.")
            quit()

        self.hidden_size = config.hidden_size if "hidden_size" in config else 32
        self.attention_hidden_size = config.attention_hidden_size if "attention_hidden_size" in config else 32
        self.attentionInputSize = self.hidden_size*2+config.embedding_dim

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        if vectors is not None:
            self.embeddings.weight.data.copy_(vectors)
            self.embeddings.weight.requires_grad = config.finetune_embeddings


        self.linear = nn.Linear(self.hidden_size*2, len(config.classes))# times 2 hidden size because of bidirectional LSTM

        self.lstm = torch.nn.LSTM(config.embedding_dim,self.hidden_size,1, batch_first=True,bidirectional=True)

        self.wordAttentionBlock = AttentionBlock(self.config, self.config.embedding_dim , 32, logger=self.logger)

        self.logger.info("Read2VecWordAttentionOnly initialized.")





    def forward(self, inputs):
        words, nSents, nWords = inputs.text

        wordMask = (words!=self.config.paddingIdx)
        sentenceMask = (nWords!=0)
        #print("words",words)
        #print("nSents",nSents,nSents.size())
        #print("nWords",nWords,nWords.size())
        #print(sentenceMask)
        #print(wordMask)

        #assert False
        embeds = self.embeddings(words)
        allSentsFlat = embeds.view(-1,embeds.size(2),embeds.size(3))
        nWordsFlat=nWords.view(-1)

        #Converting all 0 length sentences to length of 1
        #This is a hack to avoid the fact that pack_padded_sequence does not accept length 0 sequences
        #This hack should not be needed in the future as community of pytorch is working towards this feature:
        #https://github.com/pytorch/pytorch/issues/9681

        nWordsFlatNonZero= torch.Tensor.clone(nWordsFlat)
        nWordsFlatNonZero[nWordsFlatNonZero==0]=1



        order=np.argsort(-nWordsFlat)
        orderInverse=np.argsort(order)

        pack = nn.utils.rnn.pack_padded_sequence(allSentsFlat[order], nWordsFlatNonZero[order],batch_first=True)
        output, hn=self.lstm(pack )
        hAllWordsFlat, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        hAllWordsFlat = hAllWordsFlat[orderInverse]


        #Converting all h of length 0 sentences to zeros, so that the hack (see above) does not affect the end result by any means
        hAllWordsFlat[nWordsFlat==0]=0 #the 0 gets broadcasted to fit the dimensions needed

        #Revert flattening back to original hierachy by splitting using number of sentences per text
        hAllWords = hAllWordsFlat.split(embeds.size(1),dim=0)
        hAllWords= torch.stack(hAllWords,dim=0) # this step is needed as split returns a tuple

        wordAttentions = self.wordAttentionBlock(embeds,mask=wordMask)
        #we cannot use torch.mean as it would also consider padded (zero valued) words
        #therefore, sum and divide by actual length
        hAllSent = (hAllWords*wordAttentions).sum(2)

        #To ensure that sentences that are empty do not produce a NaN value.
        #h for those sentences is already a vector of zeroes, so this has no other impact on the result
        #nWordsNonZero = torch.Tensor.clone(nWords)
        #nWordsNonZero[nWordsNonZero==0] = 1

        #hAllSentAvg = hAllSentSum/(nWordsNonZero).float().unsqueeze(2).expand_as(hAllSentSum)


        #Same here, we need to compute mean based on actual number of sentences, not padded length
        hTextSum = hAllSent.sum(1)
        #same as above, there might be texts with 0 sentences as input
        nSentsNonZero = torch.Tensor.clone(nSents)
        nSentsNonZero[nSentsNonZero==0] = 1
        hTextAvg = hTextSum/(nSentsNonZero).float().unsqueeze(1).expand_as(hTextSum)


        out = self.linear(hTextAvg)

        return torch.softmax(out,dim=1)
