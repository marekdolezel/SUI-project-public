#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:40:14 2021

@author: Vlastimil Radsetoulal
"""

import numpy as np
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import process_pickle_to_np_array as pitonp
import h5py

#Neural network script, will initialize NN, train it, save it, take data from h5 file
# or the data can be rather passed by some other already complete script
# then -> model will be called for evaluation of each state of MaxN 

#Questions:
#TODO - ValueError: not enough values to unpack (expected 2, got 1)
#error line 63 - check dataloader / data provided
#TODO epochs iteration (one simple for)
#TODO check data with tainted

    
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()
transform = transforms.Compose([transforms.ToTensor()])

# TODO supervised learning NN with outputs - WIN/LOSS inputs -> vector of numbers from gameSerialize
class NetworkSui(nn.Module):
    
    def __init__(self):
        super(NetworkSui, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(663, 512),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


    def train_model(model, dataloader):
        model.train()
        size = len(dataloader.dataset)
        
        for batch, (X, y) in enumerate(dataloader, 0):
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = nn.CrossEntropyLoss(pred, y)
            
            nn.optimizer.zero_grad()
            loss.backward()
            nn.optimizer.step()
            
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
     
        
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
#extract and process gameStates files (to inputs and outputs)
    def extract_states():
        
        h5f = h5py.File('processed667files.h5','r')

        data_array = h5f['data'][:]
        h5f.close()

        data = transform(data_array)
        data = DataLoader(data, batch_size=64)

        return data
    
    def save_model(model):
        torch.save(model.state_dict(), "model.pth")             #change dir TODO
        print("Modal has been saved as model.pth")


    def load_model():
        model = NetworkSui()
        model.load_state_dict(torch.load("model.pth"))          #check dir ...
        
#for testing, might be removed entirely later        
if __name__ == "__main__":
    data = NetworkSui.extract_states()
    model = NetworkSui().to(device)

    NetworkSui.train_model(model, data)
    NetworkSui.save_model(model)