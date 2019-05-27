import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

# import your classes here

#from tensorboardX import SummaryWriter
#from utils.metrics import AverageMeter, AverageMeterList
#from utils.misc import print_cuda_statistics
from datasets.singleLanguage import SingleLanguageDataLoader
from datasets.singleLanguageSyntaxnet import SingleLanguageSyntaxnetDataLoader

cudnn.benchmark = True
from graphs.models.BaselineTwoFCLayers import BaselineTwoFCLayers
from graphs.models.BaselineLSTMandFC import BaselineLSTMandFC
from graphs.models.Read2VecWordAttention import Read2VecWordAttention
from graphs.models.Read2VecMultiAttention import Read2VecMultiAttention



from graphs.losses.cross_entropy import   CrossEntropyLoss

class Read2VecAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("Initializing Read2VecAgent...")
        self.config= config

        if "useGPU" in self.config and self.config.useGPU and torch.cuda.is_available():
            self.config.device= torch.device("cuda")
            self.logger.info("Using CUDA as device...")
        else:
            self.config.device= torch.device("cpu")
            self.logger.info("Using CPU as device...")
        #This is a fake vector so that the cluster does not kick us (the cluster has a )
        if "useGPU" in self.config and self.config.useGPU:
            self.fakeTensor = torch.cuda.FloatTensor(10, 10).fill_(0)
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
        self.data_loader = globals()[config.data_loader](config=config)


        #using only embeddings that are appearing on the dataset

        self.model = globals()[config.model](config= config,vectors = self.data_loader.TEXT.vocab.vectors, logger= self.logger)
        self.loss = globals()[config.loss](config=config)

        self.model = self.model.to(self.config.device)
        # define loss
        #self.loss = nn.NLLLoss()

        # define optimizer
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = config.learning_rate)
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        self.logger.info("Read2VecAgent initialized.")


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """

        #for batch in self.data_loader.train_iter:
        #    print(batch.text[2])
        #    print(batch.text_POS[2])
        #    print(batch.label)
        #    print()
            #break
        #print(self.data_loader.TEXT_POS.vocab.stoi)

        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """

        for epoch in tqdm(range(1, self.config.max_epoch + 1)):
            self.train_one_epoch()
            if epoch % self.config.validate_every ==0:
                self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        #self.logger.info("Train one epoch")
        totalError = 0
        totalIter = 0
        correct=0
        total=0
        for batch in (self.data_loader.train_iter):
                torch.cuda.empty_cache()
            #try:
                y = batch.label-1 # all indexes are one unit higher than what need to be, given that torchtext generates an index for the unknown token
                y= y.to(self.config.device)
                y_pred= self.model(batch)
                #print(y_pred,y-1)
                #print(y_pred,y)
                error = self.loss(y_pred,y)
                self.optimizer.zero_grad()
                error.backward()
                #self.logger.info("Error: " + str(error.data.tolist()))
                self.optimizer.step()
                totalError= totalError + error.data.tolist()
                totalIter = totalIter+1
                correct=correct+(torch.argmax(y_pred,1)==y).sum().float().cpu().data.numpy()
                total=total + y.size()[0]

            #except:
            #    self.logger.warning("Batch failed, skipping, consider checking whether there are empty files in your dataset.")
            #pass
        if totalIter >0:
            self.logger.info(" Train error (epoch "+str(self.current_epoch)+") : " + str(totalError/totalIter)+ " accuracy: "+str(correct/total))
            #print(self.accuracy(y,y_pred))

            #print(y, y_pred)

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        totalError = 0
        totalIter = 0
        correct=0
        total=0
        for batchI, batch in enumerate((self.data_loader.valid_iter)):

            #try:
                y = batch.label-1 # all indexes are one unit higher than what need to be, given that torchtext generates an index for the unknown token
                y= y.to(self.config.device)
                y_pred= self.model(batch)

                error = self.loss(y_pred,y)

                totalError= totalError + error.data.tolist()
                totalIter = totalIter+1
                correct=correct+(torch.argmax(y_pred,1)==y).sum().float().cpu().data.numpy()
                total=total + y.size()[0]
                self.createVisualization(batch, batchI)
            #except:
            #    self.logger.warning("Batch failed, skipping, consider checking whether there are empty files in your dataset.")
            #pass
        if totalIter >0:
            self.logger.info(" Validation error (epoch "+str(self.current_epoch)+") : " +  str(totalError/totalIter)+ " accuracy: "+str(correct/total))



    def createVisualization(self,batch,batchI):
        outputDir = "experiments/{}/out/".format(self.config.exp_name)
        for i in range(self.config.batch_size):
            filename= "attention_epoch{}_batch{}_i{}.html".format(self.current_epoch,batchI,i)
            html=""
            if(i==0):
                html= html + "<html><body>"
                html= html + "<h1>Word Level Semantic</h1>"
                html = html + "<p>"
                words, nSents, nWords = batch.text
                wordAtt, sentAtt = self.model.attention(batch)
                maxAtt = torch.max(wordAtt)
                minAtt = torch.min(wordAtt)
                html = html + "<p>"
                for si in range(nSents[0]):
                    for wi in range(nWords[0][si]):
                        #idx = words[0][si][wi]
                        watt= wordAtt[0][si][wi].cpu().data.numpy()[0]
                        watt=(watt-minAtt)/(maxAtt-minAtt)
                        watt= 256-(256*watt)
                        wform =self.data_loader.TEXT.vocab.itos[words[0][si][wi]]
                        try:
                            html = html + "<span style=\"background-color:rgb(256,{},{});\">{}</span> ".format(watt,watt,wform)
                        except:
                            pass
                html = html + "</p>"
                html = html + "</body></html>"
                #print(html)

                with open(outputDir+filename,"w", encoding="utf-8") as fout:
                    fout.write(html)
                #print(filename)
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass

    def accuracy(self,y,y_pred):
        return (torch.argmax(y_pred,1)==y).sum().float().cpu().data.numpy()/y.size()[0]
