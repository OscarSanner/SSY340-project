import torch
from torch import nn, optim
from .loss import GANLoss


class EnsembleHeadColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3),
            nn.MaxPool2d(),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv1(x)