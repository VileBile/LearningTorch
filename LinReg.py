#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:02:03 2022

@author: marko
"""
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt 


                            #From Scratch#
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

def gen_wb(inputs,targets):
    ix, iy = inputs.shape
    tx, ty = targets.shape
    W = torch.rand(ty,iy, requires_grad=True)
    b = torch.rand(ty, requires_grad=True)
    return W, b

def mse(o,t):
    diff = o-t
    return torch.sum(diff*diff)/diff.numel()


def linreg(inputs,targets,its=1000,lr=10**(-4)):
    W, b = gen_wb(inputs,targets)
    L = []
    for i in range(its):
        outs = inputs @ W.t() + b
        loss = mse(outs,targets)
        L.append(loss.item())
        loss.backward()
        with torch.no_grad():
            W -= lr*W.grad
            b -= lr*b.grad
            W.grad.zero_()
            b.grad.zero_()
        
    return W, b, L 

                            #Using Torch built-ins#
                            
model = nn.Linear(3,2)
DataSet = TensorDataset(inputs,targets)
bs = 5
dl = DataLoader(DataSet,batch_size=bs,shuffle=True)
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(),lr=10**-4)

def fit(epochs,model,loss_fn,opt,dataloader):
    L = []
    for i in range(epochs):
        for x, y in dataloader:
            preds = model(x)
            loss = loss_fn(preds,y)
            L.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
            
    return model, L        
            
            
            
            
            
    
