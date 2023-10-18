import torch
from torch import nn, optim

class EnsembleHeadColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        res = self.layers(x)
        # Input x: torch.Size([16, 12, 256, 256])
        # Output res: torch.Size([16, 3, 256, 256])
        return res