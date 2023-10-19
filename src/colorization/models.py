import torch
from torch import nn, optim

class EnsembleHeadColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        res = self.layers(x)
        # Input x: torch.Size([16, 12, 256, 256])
        # Output res: torch.Size([16, 3, 256, 256])
        return res

class MeanChannelColorizer():
    def forward(self, x):
        # Input x: torch.Size([16, 12, 256, 256])
        # Output res: torch.Size([16, 3, 256, 256])
        # Split the tensor by colors
        reds = x[[0, 3, 6, 9], :, :]
        greens = x[[1, 4, 7, 10], :, :]
        blues = x[[2, 5, 8, 11], :, :]

        avg_reds = reds.mean(dim=0, keepdim=True)
        avg_greens = greens.mean(dim=0, keepdim=True)
        avg_blues = blues.mean(dim=0, keepdim=True)

        return torch.cat([avg_reds, avg_greens, avg_blues], dim=0)