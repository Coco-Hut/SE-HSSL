import torch
import torch.nn as nn

class LogReg(nn.Module):
    
    def __init__(self, num_features, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z