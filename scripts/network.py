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
from torch.utils.data import TensorDataset
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
#loss_fn = torch.nn.MSELoss
#loss_fn = nn.KLDivLoss()

transform = transforms.Compose([transforms.ToTensor()])

# TODO supervised learning NN with outputs - WIN/LOSS inputs -> vector of numbers from gameSerialize
class NetworkSui(nn.Module):
    
    def __init__(self):
        super(NetworkSui, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(663, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


    def train_model(model, dataloader, optimizer):
        model.train()
        size = len(dataloader.dataset)
        print(size)
        
        for batch, (X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            #print(pred)
            #print(y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
     
        
    def test(model, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                test_loss += loss_fn(pred, y).item()

                correct += (pred.argmax(0) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
      
# 1-hot encodes a tensor        
    def to_categorical(y, num_classes):
        y = y.astype(int)
        
        labels = np.zeros(shape=(y.shape[0], num_classes))

        i = 0
        for row in y:
            
            labels[i][(y[i])-1] = 1
            i += 1

        return labels
        
#extract and process gameStates files (to data and labels) returns dataloader object
    def extract_states():
        
        h5f = h5py.File('UniformDataset200K.h5','r')

        data_array = h5f['data'][:]
        h5f.close()


        data = np.zeros(shape=(data_array.shape[0], 663))
        labels = np.zeros(shape=(data_array.shape[0], 1))
        

        i = 0
        for row in data_array:
            data[i] = np.copy(data_array[i][:-1])           
            labels[i] = np.copy(data_array[i][-1])
            i += 1

        
        data = transform(data)


        labels = NetworkSui.to_categorical(labels, 4)        
        labels = transform(labels)

        data = torch.squeeze(data, dim=0)
        labels = torch.squeeze(labels, dim = 0)
        
        #print(data.shape)
        
        dataset = TensorDataset(data, labels)
        train_set, val_set = torch.utils.data.random_split(dataset, [180000,20000])
        data = DataLoader(train_set, batch_size=64)
        test_data = DataLoader(val_set, batch_size=64)

        return data, test_data
    
    
    def iterate_epochs(model, epochs, train_dataloader, val_dataloader, optimizer):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            NetworkSui.train_model(model, train_dataloader, optimizer)
            NetworkSui.test(model, val_dataloader)
            print("Done!")
    
    def save_model(model):
        torch.save(model.state_dict(), "model.pth")             #change dir TODO
        print("Model has been saved as model.pth")


    def load_model():
        model = NetworkSui()
        model.load_state_dict(torch.load("model.pth"))          #check dir ...
        
#for testing, might be removed entirely later        
if __name__ == "__main__":
    data, test_data = NetworkSui.extract_states()
    print(len(data))

    model = NetworkSui().to(device)
    model.double()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0, amsgrad=False)

    NetworkSui.iterate_epochs(model, 25, data, test_data, optimizer)
    NetworkSui.save_model(model)