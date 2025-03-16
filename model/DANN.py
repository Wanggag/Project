import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import torch.nn.functional as F
from torch.autograd import Function

import matplotlib.pyplot as plt
import scipy.io as sio
import pprint

cuda =torch.cuda.is_available()
if cuda:
    device='cuda'
else:
    device='cpu'

class Extractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout):
        super(Extractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = num_layers
        self.dropout = dropout
        ##特征提取器____lstm
        self.lstm = nn.LSTM(50, 
                            25, 
                            self.layer_size,
                            dropout = self.dropout,
                            batch_first= False
        )  
        # self.lstm2 = nn.LSTM(self.hidden_size, 
        #             100, 
        #             self.layer_size,
        #             dropout = self.dropout,
        #             )        

    def forward(self, x):
        h0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        #初始化细胞状态
        c0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        
        lstm_out,(hn,cn) = self.lstm(x)
        # lstm_out,(hn,cn) = self.lstm2(lstm_out1)

        return lstm_out
#%%域判别器    
class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

class dann(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, dropout ):
        super(dann, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_size = num_layers
        ##特征提取器____lstm
        self.lstm = nn.LSTM(self.input_size, 
                            self.hidden_size, 
                            self.layer_size,
                            dropout=self.dropout,
                            batch_first= True
        )  
        self.fcRelu2 = nn.Sequential(
            nn.Linear(self.hidden_size, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
        )
        self.dc = nn.Sequential(nn.Linear(100, 2))
        
    def forward(self, x,alpha):
        h0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        #初始化细胞状态
        c0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        lstm_out,(hn,cn) = self.lstm(x)
        grl_out = GRL.apply(lstm_out[:,-1,:], alpha)
        domainClass =self.dc(grl_out) 
        return lstm_out,grl_out,domainClass



class Dann(nn.Module):
    def __init__(self, ):
        super(Dann, self).__init__()
        
        #全连接+relu层，共享的层
        self.fcRelu2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
        )       
        ##    域分类器
        self.dc = nn.Sequential(nn.Linear(25, 3))
        
    def forward(self, x,alpha): 
        grl_out = GRL.apply(x, alpha)
        gf_out = self.fcRelu2(grl_out[:,-1,:]) 
        domainClass =self.dc(gf_out)  
                       
        return gf_out, domainClass
 #%%域判别器    
class Regression1(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, dropout):
        super(Regression1, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_size = num_layers
        ##特征提取器____lstm
        self.lstm = nn.LSTM(25, 
                            100, 
                            3,
                            dropout = self.dropout,
                            batch_first= True
        )  
        self.fcRelu2 = nn.Sequential(
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
        )
        self.Rc =  nn.Linear(25, 1)
       
    def forward(self, x):
        h0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        #初始化细胞状态
        c0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        lstm_out,(hn,cn) = self.lstm(x)
        gf_out = self.fcRelu2(lstm_out[:,-1,:])  
        rul =self.Rc(gf_out)

        return lstm_out,gf_out,rul

class Regression2(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, dropout ):
        super(Regression2, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_size = num_layers
        ##特征提取器____lstm
        self.lstm = nn.LSTM(25, 
                            100, 
                            3,
                            dropout = self.dropout,
                            batch_first= False
        )  
        self.fcRelu2 = nn.Sequential(
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
        )
        self.Rc =  nn.Linear(25, 1)
        ##    域分类器
        self.dc = nn.Sequential(nn.Linear(25, 2))
        
    def forward(self, x,alpha):
        h0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        #初始化细胞状态
        c0 = torch.zeros(self.layer_size, x.size(1), self.hidden_size).requires_grad_().to(device)
        lstm_out,(hn,cn) = self.lstm(x)
        gf_out = self.fcRelu2(lstm_out[:,-1,:])  
        grl_out = GRL.apply(gf_out, alpha)
        rul =self.Rc(gf_out)
        domainClass =self.dc(grl_out) 

        return lstm_out,gf_out,rul,domainClass
