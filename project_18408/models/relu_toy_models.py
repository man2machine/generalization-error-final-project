# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:27:02 2021

@author: Shahir
"""

import torch.nn as nn
import torch.nn.functional as F

class ReLUToyModel(nn.Module):
    def __init__(self,
                 input_dim,
                 layer_dims=(100, 100),
                 num_classes=10):
        super().__init__()
        layers = []
        for output_dim in layer_dims:
            layers.append(nn.Linear(input_dim, output_dim, bias=True))
            layers.append(nn.ReLU(inplace=True))
            input_dim = output_dim
        self.layers.append(nn.Linear(input_dim, num_classes))
        self.layers = nn.Sequential(*layers)
    
    def _initialize_weights(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.layers(x)
        return x
