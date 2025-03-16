# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

cuda =torch.cuda.is_available()
if cuda:
    device='cuda'
else:
    device='cpu'

class LSTMmodel1(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers,dropout):
        super(LSTMmodel, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers , batch_first=True)
        
        linear_list = [100,50,25]
        linear_seq = []
        for i in range(0, len(linear_list)-1):
            linear_seq = linear_seq + self.linear_block(
                linear_list[i], linear_list[i+1])
        self.linear_Layer = nn.Sequential(*linear_seq)
        self.linear_out = nn.Linear(linear_list[-1], self.output_size)
        
        # self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def linear_block(self, in_features, out_features):
        block  =  [nn.Linear(in_features, out_features)]
        block +=  [nn.BatchNorm1d(out_features)]
        block +=  [nn.ReLU()]
        
        return block
    def forward(self, X, future=0):
        
        out0, (h0, c0) = self.lstm(X)
        outh = self.linear_Layer(out0[:,-1,:])
        # denseh=torch.cat([outh,hhx],1)
        output=self.linear_out(outh)

        return out0,outh,output
class LSTMmodel(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers,dropout):
        super(LSTMmodel, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers 
        self.dropout=dropout 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers ,dropout=self.dropout, batch_first=True)
        
        linear_list = [100,50,25]
        linear_seq = []
        for i in range(0, len(linear_list)-1):
            linear_seq = linear_seq + self.linear_block(
                linear_list[i], linear_list[i+1])
        self.linear_Layer = nn.Sequential(*linear_seq)
        self.linear_out = nn.Linear(linear_list[-1], self.output_size)
        
        # self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def linear_block(self, in_features, out_features):
        block  =  [nn.Linear(in_features, out_features)]
        block +=  [nn.BatchNorm1d(out_features)]
        block +=  [nn.ReLU()]
        return block
    
    def forward(self, X, future=0):
        
        out0, (h0, c0) = self.lstm(X)
        outh = self.linear_Layer(out0[:,-1,:])
        output=self.linear_out(outh)

        return out0,outh,output
class LSTM3(nn.Module):
    def __init__(self, ):
        super(LSTM3, self).__init__()
        self.rul = nn.Sequential(nn.Linear(25, 1))
        
    def forward(self, x): 
        # gf_out = self.fcRelu2(x)
        # grl_out = GRL.apply(gf_out, alpha)
        # domainClass =self.dc(grl_out)  
        rul = self.rul(x)
                       
        return rul
    



#%%域判别器    
class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

class domain2(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(domain2, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers ,batch_first=True)
        
        linear_list = [100]
        linear_seq = []
        for i in range(0, len(linear_list)-1):
            linear_seq = linear_seq + self.linear_block(
                linear_list[i], linear_list[i+1])
        self.linear_Layer = nn.Sequential(*linear_seq)
        self.linear_out = nn.Linear(linear_list[-1], self.output_size)
        
        # self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def linear_block(self, in_features, out_features):
        block  =  [nn.Linear(in_features, out_features)]
        block +=  [nn.BatchNorm1d(out_features)]
        block +=  [nn.ReLU()]
        return block
    
    def forward(self, X,alpha, future=0):
        
        out0, (h0, c0) = self.lstm(X)
        outh = self.linear_Layer(out0[:,-1,:])
        grl_out = GRL.apply(outh, alpha)
        # denseh=torch.cat([outh,hhx],1)
        domainClass=self.linear_out(grl_out)

        return out0,grl_out,domainClass


class LSTMmodel2(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(LSTMmodel2, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)
        
        linear_list = [100]
        linear_seq = []
        for i in range(0, len(linear_list)-1):
            linear_seq = linear_seq + self.linear_block(
                linear_list[i], linear_list[i+1])
        self.linear_Layer = nn.Sequential(*linear_seq)
        self.linear_out = nn.Linear(linear_list[-1], self.output_size)
        
        # self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def linear_block(self, in_features, out_features):
        block  =  [nn.Linear(in_features, out_features)]
        block +=  [nn.BatchNorm1d(out_features)]
        block +=  [nn.ReLU()]
        return block
    
    def forward(self, X, future=0):
        
        out0, (h0, c0) = self.lstm(X)
        outh = self.linear_Layer(out0[:,-1,:])
        # denseh=torch.cat([outh,hhx],1)
        output=self.linear_out(outh)

        return out0,outh,output

class Sequence(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers,time_steps):
        super(Sequence, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size#hidden_size即为输出维度
        self.output_size=output_size
        self.num_layers=num_layers 
        self.time_steps=time_steps
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers)
        
        # self.lstm1 = nn.LSTM(self.hidden_size,self.hidden_size,self.num_layers)
        
        self.linear = nn.Linear(self.hidden_size,self.output_size)

    def forward(self, X):

        h0 = torch.zeros(self.num_layers,self.time_steps,self.hidden_size , dtype=torch.double).to(device)
        c0 = torch.zeros(self.num_layers,self.time_steps,self.hidden_size , dtype=torch.double).to(device)
        out1,(h0, c0) = self.lstm(X,(h0, c0))
        # out,(_,_) = self.lstm1(out1,(h0, c0))
        output = self.linear(out1)

        return out1,out1[:,-1,:],output[:,-1,:]

#%%域判别器    
class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

class domain(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers,time_steps):
        super(domain, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size#hidden_size即为输出维度
        self.output_size=output_size
        self.num_layers=num_layers 
        self.time_steps=time_steps
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers)
        
        # self.lstm1 = nn.LSTM(self.hidden_size,self.hidden_size,self.num_layers)
        
        self.linear = nn.Linear(self.hidden_size,self.output_size)

    def forward(self, X,alpha):

        h0 = torch.zeros(self.num_layers,self.time_steps,self.hidden_size , dtype=torch.double).to(device)
        c0 = torch.zeros(self.num_layers,self.time_steps,self.hidden_size , dtype=torch.double).to(device)
        out1,(h0, c0) = self.lstm(X,(h0, c0))
        # out,(_,_) = self.lstm1(out1,(h0, c0))
        grl_out = GRL.apply(out1, alpha)
        domainClass = self.linear(grl_out)

        return out1,grl_out[:,-1,:],domainClass[:,-1,:]
