import torch
import torch.nn as nn
import torch.nn.functional as F

class Layernorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels)
      
    def forward(self, x):
        bn, c, h, w = x.shape
        x = x.view(bn, c, -1).transpose(1, 2)
        x = self.layernorm(x).transpose(2, 1)
        return x.view(bn, c, h, w)

class RB(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding="same"), Layernorm(out_c), nn.GELU())
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)
      
    def forward(self, x):
        return self.net(x) + self.skip(x)

class MLP(nn.Module):
    def __init__(self, dim, embedding_size=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, embedding_size))
      
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1)) + F.adaptive_max_pool2d(x, (1, 1))
        return self.net(x.view(x.shape[0], -1))

class EMA():
    def __init__(self, beta): self.beta = beta
    def update_average(self, old, new): return old * self.beta + (1 - self.beta) * new
