import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleModel(nn.Module):
    def __init__(self,input_dim=20, output_dim=5):
        super(simpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim,64)
        self.norm1 = nn.BatchNorm1d(64)
        self.Relu = nn.ReLU()
        self.layer2 = nn.Linear(64,128)
        self.norm2 = nn.BatchNorm1d(128)
        self.liner1 = nn.Linear(128,64)
        self.liner2 = nn.Linear(64,32)
        self.liner3 = nn.Linear(32,output_dim)
      
    def forward(self,x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.Relu(x)
        x = self.layer2(x)
        x = self.norm2(x)
        x = self.Relu(x)
        x= self.liner1(x)
        x = self.Relu(x)
        x = self.liner2(x)
        x = self.Relu(x)
        x = self.liner3(x)

        return x

