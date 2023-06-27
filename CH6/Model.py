import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, unpacked_params):
        super().__init__()
        self.l1, self.b1, self.l2, self.b2, self.l3, self.b3 = unpacked_params
        self.relu = F.relu
    
    def forward(self, x):
        out = F.linear(x, self.l1, self.b1)
        out = self.relu(out)
        out = F.linear(out, self.l2, self.b2)
        out = self.relu(out)
        out = F.linear(out, self.l3, self.b3)
        out = torch.log_softmax(out, dim=0)
        
        return out