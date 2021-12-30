#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:40:14 2021

@author: Vlastimil Radsetoulal
"""
from matplotlib import pyplot as plt
import numpy as np
import pickle
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import process_pickle_to_np_array as pitonp

#Neural network script, will initialize NN, train it, save it, take data from npy file
# or the data can be rather passed by some other already complete script
# then -> model will be called for evaluation of each state of MaxN 

#Warning -> this needs gameStates file to be in scripts. After model is trained this script won't need to be launched TODO

#Questions:
    # Do we have data of each state or only of initiate states ?
    # Do players always have the same numbers ? If not we need to asure it
    # Do we know in the adjucency matrice always determine which player we are ? 
    # is it adjucency of our player/AI ? If yes - that's great !



# TODO supervised learning NN with outputs - WIN/LOSS inputs -> vector of numbers from gameSerialize
class NetworkSui(nn.Module):
    
    def __init__(self):
        super(NetworkSui, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(663, 330),
            nn.ReLU(),
            nn.Linear(330, 165),
            nn.ReLU(),
            nn.Linear(165, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#extract and process gameStates files (to inputs and outputs)
    def extract_states():
        
        data_array = np.load("gameStates/1230_2021_195436.npy")      #change this - only processess one file TODO
        
        outputs = np.zeros((data_array.shape[0],1))
        inputs = np.zeros((data_array.shape[0],663))
        
        index = 0
        for row in data_array:
            outputs[index] = row[-1] if row[-1]==1 else 0       #1 if player #1 wins otherwise 0 (lost) which number is our AI ? TODO
            inputs[index] = data_array[index][:-1]
            index+=1
            

        data = (inputs, outputs)
        return data
    
    
    #fit model with inputs outputs, iterate epochs TODO
    def train_model():
        return 0
        
#for testing, might be removed entirely later        
if __name__ == "__main__":
    data_tuple = NetworkSui.extract_states()
    #print(data_tuple)                                           check how data tuple looks if you want
