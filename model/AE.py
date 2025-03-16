import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import scipy.io as sio
import pprint

cuda =torch.cuda.is_available()
if cuda:
    device='cuda'
else:
    device='cpu'

class AutoEncoder11(nn.Module,):
    def __init__(self, n_in, encoder_units):
        super(AutoEncoder, self).__init__()
        
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.in_size=n_in
        self.encoder = nn.Sequential(
            nn.Linear(self.in_size,self.encoder_units[0]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[0], self.encoder_units[1]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[1], self.encoder_units[2]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[2], self.encoder_units[3]),   # compress to 3 features which can be visualized in plt
            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_units[0], self.decoder_units[1]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[1], self.decoder_units[2]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[2], self.decoder_units[3]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[3], self.in_size),
            
        )

    def forward(self,x):
        en = self.encoder(x)
        de = self.decoder(en)
        return en,de

class AutoEncoder(nn.Module,):
    def __init__(self, n_in, encoder_units):
        super(AutoEncoder, self).__init__()
        
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.in_size=n_in
        self.encoder = nn.Sequential(
            nn.Linear(self.in_size,self.encoder_units[0]),      
            nn.ReLU(),
            nn.Linear(self.encoder_units[0], self.encoder_units[1]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[1], self.encoder_units[2]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[2], self.encoder_units[3]),
            nn.BatchNorm1d(self.encoder_units[3]),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_units[0], self.decoder_units[1]),
            nn.BatchNorm1d(self.decoder_units[1]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[1], self.decoder_units[2]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[2], self.decoder_units[3]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[3], self.in_size),
        )

    def forward(self,x):
        en = self.encoder(x)
        de = self.decoder(en)
        return en,de
