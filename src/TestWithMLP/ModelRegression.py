import torch
import torch.nn as nn

class MLPR(nn.Module):
    def __init__(self, size_input=20):
        super().__init__()
        self.regressionHead = nn.Sequential(
            nn.Linear(size_input,64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32,1))
    def forward(self, input):
        pred = self.regressionHead(input)
        return input