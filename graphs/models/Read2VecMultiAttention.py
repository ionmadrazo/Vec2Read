
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from graphs.models.custom_layers.AttentionBlock import AttentionBlock
from graphs.models.custom_layers.EvenlyDistributedAttentionBlock import EvenlyDistributedAttentionBlock
from graphs.models.custom_layers.MultiAttentionBlock import MultiAttentionBlock
from graphs.models.custom_layers.VariableLengthLSTM import VariableLengthLSTM


class Read2VecMultiAttention(nn.Module):

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

        self.embeddingsPOS = nn.Embedding(config.vocab_size_pos, config.embeddingPOS_dim)
        self.embeddingsMorph = nn.Embedding(config.vocab_size_morph, config.embeddingMorph_dim)


        self.linear = nn.Linear(self.hidden_size*2, len(config.classes))# times 2 hidden size because of bidirectional LSTM

        #self.lstm = torch.nn.LSTM(config.embedding_dim,self.hidden_size,1, batch_first=True,bidirectional=True)
        self.wordLSTM = VariableLengthLSTM(config.embedding_dim,self.hidden_size,1, bidirectional=True)
        self.wordLSTM.to(self.config.device)
        if "wordS" in config.attention2input_size:
            assert int(config.attention2input_size["wordS"]/2) == self.hidden_size


        if "synS" in config.attention2input_size:
            self.synLSTM = VariableLengthLSTM(config.embeddingPOS_dim,int(config.attention2input_size["synS"]/2),1, bidirectional=True)
            self.synLSTM.to(self.config.device)
        if "morphS" in config.attention2input_size:
            self.morphLSTM = VariableLengthLSTM(config.embeddingMorph_dim,int(config.attention2input_size["morphS"]/2),1, bidirectional=True)
            self.morphLSTM.to(self.config.device)
        assert len(config.attention2hidden_size) == len(config.attention2input_size)
        attBlockDictW={}
        attBlockDictS={}
        possibleAttentions = ["wordW","synW","morphW","wordS","synS","morphS"]
        usedAttentions= []
        for attName in possibleAttentions:
            if attName in config.attention2hidden_size and attName.endswith("W"):
                 attBlockDictW[attName] = AttentionBlock(self.config, config.attention2input_size[attName] , config.attention2hidden_size[attName], logger=self.logger)
                 attBlockDictW[attName].to(self.config.device)
            elif attName in config.attention2hidden_size and attName.endswith("S"):
                 attBlockDictS[attName] = AttentionBlock(self.config, config.attention2input_size[attName] , config.attention2hidden_size[attName], logger=self.logger)
                 attBlockDictS[attName].to(self.config.device)
            if attName in config.attention2hidden_size:
                usedAttentions.append(attName)

        self.logger.info("Using the following attention blocks: "+ str(usedAttentions))
        #self.synAttentionBlockWord = AttentionBlock(self.config, self.config.embeddingPOS_dim , self.config.attention_hidden_size_syn, logger=self.logger)
        #self.morphAttentionBlockWord = AttentionBlock(self.config, self.config.embeddingMorph_dim , self.config.attention_hidden_size_morph, logger=self.logger)
        #attBlockDict= {"word":self.wordAttentionBlockWord,
        #                "syn":self.synAttentionBlockWord,
        #                "morph":self.morphAttentionBlockWord}
        #attBlockDict= {"word":self.wordAttentionBlock}





        #self.wordAttentionBlockSent = AttentionBlock(self.config, self.hidden_size*2 , self.config.attention_hidden_size_word, logger=self.logger)
        #self.synAttentionBlockSent = AttentionBlock(self.config, self.hidden_size*2 , self.config.attention_hidden_size_syn, logger=self.logger)
        #self.morphAttentionBlockSent = AttentionBlock(self.config, self.hidden_size*2 , self.config.attention_hidden_size_morph, logger=self.logger)
        #attBlockDict= {"word":self.wordAttentionBlockSent,
        #                "syn":self.synAttentionBlockSent,
        #                "morph":self.morphAttentionBlockSent}
        #attBlockDict= {"word":self.wordAttentionBlock}

        if len(attBlockDictW) ==0:
            attBlockDictW["even"] = EvenlyDistributedAttentionBlock(config,logger=self.logger)
            attBlockDictW["even"].to(self.config.device)
        if len(attBlockDictS) ==0:
            attBlockDictS["even"] =  EvenlyDistributedAttentionBlock(config,logger=self.logger)
            attBlockDictS["even"].to(self.config.device)
        self.wordLevelMultiattentionBlock = MultiAttentionBlock(config,attentionBlocks=attBlockDictW,logger=self.logger)
        self.sentenceLevelMultiattentionBlock = MultiAttentionBlock(config,attentionBlocks=attBlockDictS,logger=self.logger)
        self.wordLevelMultiattentionBlock.to(self.config.device)
        self.sentenceLevelMultiattentionBlock.to(self.config.device)



        self.logger.info("Read2VecWordAttentionOnly initialized.")





    def forward(self, inputs):
        words, nSents, nWords = inputs.text
        pos, _, _ = inputs.pos
        morph, _, _ = inputs.morph

        words=words.to(self.config.device)
        nSents=nSents.to(self.config.device)
        nWords=nWords.to(self.config.device)
        pos=pos.to(self.config.device)
        morph=morph.to(self.config.device)

        wordMask = (words!=self.config.paddingIdx)
        sentenceMask = (nWords!=0)
        #print("words",words)
        #print("nSents",nSents,nSents.size())
        #print("nWords",nWords,nWords.size())
        #print(sentenceMask)
        #print(wordMask)

        #assert False
        embeds = self.embeddings(words)
        embedsPOS = self.embeddingsPOS(pos)
        embedsMorph = self.embeddingsMorph(morph)


        hAllWords, hnWords = self.wordLSTM(embeds,nWords)


        inputDictW={}
        inputDictS={}

        if "wordW" in self.config.attention2hidden_size:
            inputDictW["wordW"] = embeds
        if "synW" in self.config.attention2hidden_size:
            inputDictW["synW"] = embedsPOS
        if "wordW" in self.config.attention2hidden_size:
            inputDictW["morphW"] = embedsMorph

        if "wordS" in self.config.attention2hidden_size:
            #we reuse the hnWord from main architecture
            inputDictS["wordS"] = hnWords
        if "synS" in self.config.attention2hidden_size:
            hAllSyn, hnSyn = self.synLSTM(embedsPOS,nWords)
            inputDictS["synS"] = hnSyn
        if "morphS" in self.config.attention2hidden_size:
            hAllMorph, hnMorph = self.morphLSTM(embedsMorph,nWords)
            inputDictS["morphS"] = hnMorph

        if len(inputDictW)==0:
            inputDictW["even"] = embeds
        if len(inputDictS)==0:
            inputDictS["even"] = hnWords

        wordAttentions = self.wordLevelMultiattentionBlock(inputs=inputDictW,mask=wordMask)
        sentenceAttentions = self.sentenceLevelMultiattentionBlock(inputs=inputDictS,mask=sentenceMask)
        #we cannot use torch.mean as it would also consider padded (zero valued) words
        #therefore, sum and divide by actual length
        hAllSent = (hAllWords*wordAttentions).sum(2)

        #To ensure that sentences that are empty do not produce a NaN value.
        #h for those sentences is already a vector of zeroes, so this has no other impact on the result
        #nWordsNonZero = torch.Tensor.clone(nWords)
        #nWordsNonZero[nWordsNonZero==0] = 1

        #hAllSentAvg = hAllSentSum/(nWordsNonZero).float().unsqueeze(2).expand_as(hAllSentSum)

        hText = (hAllSent*sentenceAttentions).sum(1)

        #Same here, we need to compute mean based on actual number of sentences, not padded length
        #hTextSum = (hAllSent).sum(1)
        #same as above, there might be texts with 0 sentences as input
        #nSentsNonZero = torch.Tensor.clone(nSents)
        #nSentsNonZero[nSentsNonZero==0] = 1
        #hTextAvg = hTextSum/(nSentsNonZero).float().unsqueeze(1).expand_as(hTextSum)


        out = self.linear(hText)

        return torch.softmax(out,dim=1)

    def attention(self,inputs):
        words, nSents, nWords = inputs.text
        pos, _, _ = inputs.pos
        morph, _, _ = inputs.morph

        words=words.to(self.config.device)
        nSents=nSents.to(self.config.device)
        nWords=nWords.to(self.config.device)
        pos=pos.to(self.config.device)
        morph=morph.to(self.config.device)

        wordMask = (words!=self.config.paddingIdx)
        sentenceMask = (nWords!=0)
        #print("words",words)
        #print("nSents",nSents,nSents.size())
        #print("nWords",nWords,nWords.size())
        #print(sentenceMask)
        #print(wordMask)

        #assert False
        embeds = self.embeddings(words)
        embedsPOS = self.embeddingsPOS(pos)
        embedsMorph = self.embeddingsMorph(morph)


        hAllWords, hnWords = self.wordLSTM(embeds,nWords)


        inputDictW={}
        inputDictS={}

        if "wordW" in self.config.attention2hidden_size:
            inputDictW["wordW"] = embeds
        if "synW" in self.config.attention2hidden_size:
            inputDictW["synW"] = embedsPOS
        if "wordW" in self.config.attention2hidden_size:
            inputDictW["morphW"] = embedsMorph

        if "wordS" in self.config.attention2hidden_size:
            #we reuse the hnWord from main architecture
            inputDictS["wordS"] = hnWords
        if "synS" in self.config.attention2hidden_size:
            hAllSyn, hnSyn = self.synLSTM(embedsPOS,nWords)
            inputDictS["synS"] = hnSyn
        if "morphS" in self.config.attention2hidden_size:
            hAllMorph, hnMorph = self.morphLSTM(embedsMorph,nWords)
            inputDictS["morphS"] = hnMorph

        if len(inputDictW)==0:
            inputDictW["even"] = embeds
        if len(inputDictS)==0:
            inputDictS["even"] = hnWords

        wordAttentions = self.wordLevelMultiattentionBlock(inputs=inputDictW,mask=wordMask)
        sentenceAttentions = self.sentenceLevelMultiattentionBlock(inputs=inputDictS,mask=sentenceMask)
        return wordAttentions,sentenceAttentions
