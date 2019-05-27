"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super(CrossEntropyLoss, self).__init__()
        if config is None or "class_weights"  not in config:
            self.loss = nn.CrossEntropyLoss()
        else   :

            class_weights = np.load(config.class_weights)
            self.loss = nn.CrossEntropyLoss(
                                      weight=torch.from_numpy(class_weights.astype(np.float32)),
                                      size_average=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
