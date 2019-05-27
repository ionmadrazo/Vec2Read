
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F

class VariableLengthLSTM(nn.Module):

    def __init__(self, inputDim, hiddenDim, nLayers, bidirectional  ):
        super().__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nLayers = nLayers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(inputDim,hiddenDim,nLayers, batch_first=True,bidirectional=bidirectional)



    def forward(self, embeds,nWords):

                allSentsFlat = embeds.view(-1,embeds.size(2),embeds.size(3))
                nWordsFlat=nWords.view(-1)

                #Converting all 0 length sentences to length of 1
                #This is a hack to avoid the fact that pack_padded_sequence does not accept length 0 sequences
                #This hack should not be needed in the future as community of pytorch is working towards this feature:
                #https://github.com/pytorch/pytorch/issues/9681

                nWordsFlatNonZero= torch.Tensor.clone(nWordsFlat)
                #nWordsFlatNonZero= nWordsFlatNonZero.cpu()
                #nWordsFlatNonZero= nWordsFlatNonZero.type(torch.LongTensor)
                #nWordsFlatNonZero = torch.as_tensor(nWordsFlatNonZero, dtype=torch.int64).to(torch.device("cpu"))
                nWordsFlatNonZero[nWordsFlatNonZero==0]=1



                order=np.argsort(-nWordsFlat.cpu())
                orderInverse=np.argsort(order)
                #print(order)
                #print(nWords)
                #print(nWordsFlatNonZero[order].cpu())
                #print(nWordsFlatNonZero[order].cpu().type(torch.LongTensor))
                #print(nWordsFlatNonZero.size())
                #print(nWordsFlatNonZero.dim(),nWordsFlatNonZero.dtype,nWordsFlatNonZero.device)
                #print(nWordsFlatNonZero[order].cpu().data.numpy())
                #pack = nn.utils.rnn.pack_padded_sequence(allSentsFlat[order].cpu(), nWordsFlatNonZero[order].cpu().data.numpy(),batch_first=True)
                pack = nn.utils.rnn.pack_padded_sequence(allSentsFlat[order], nWordsFlatNonZero[order],batch_first=True)

                output, (hn,cn)=self.lstm(pack )
                hAllWordsFlat, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

                hAllWordsFlat = hAllWordsFlat[orderInverse]


                #Converting all h of length 0 sentences to zeros, so that the hack (see above) does not affect the end result by any means
                hAllWordsFlat[nWordsFlat==0]=0 #the 0 gets broadcasted to fit the dimensions needed

                #Revert flattening back to original hierachy by splitting using number of sentences per text
                hAllWords = hAllWordsFlat.split(embeds.size(1),dim=0)
                hAllWords= torch.stack(hAllWords,dim=0) # this step is needed as split returns a tuple

                #Unflatten and prepare hn
                hn =torch.cat([hn[i] for i in range(hn.size(0))], 1)
                hn = hn.split(embeds.size(1),dim=0)
                hn= torch.stack(hn,dim=0)

                return hAllWords, hn
