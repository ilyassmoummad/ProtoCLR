import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
class Standardization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / self.std
    
class Projector(nn.Module):
    def __init__(self, model_name='cvt', out_dim=128):
        super(Projector, self).__init__()
        dim = model_dim[model_name]
        out_dim = out_dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(dim, out_dim))

    def forward(self, features):
        return self.mlp(features)
    
model_dim = {
    'cvt': 384,
}