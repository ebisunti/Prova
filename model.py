import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    #define the structure of the neural network
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,32)
        self.out = nn.Linear(32, 4)
        
    def forward(self, state):
        x0 = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x0))
        x2 = F.relu(self.fc3(x1))
        x3 = self.out(x2)
        return x3
