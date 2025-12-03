import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, size_input=20):
        super().__init__()
        self.classificasionHead = nn.Sequential(
            nn.Linear(size_input, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,3)

        )

    def forward(self, input):
        logits = self.classificasionHead(input)
        return logits
       
            
        
    