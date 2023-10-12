import torch
import torch.nn.functional as F
from torch import nn, reshape, optim
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, ColorJitter, RandomRotation
from torch.utils.data import DataLoader, Dataset, random_split
from itertools import chain
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

torch.manual_seed(0)
np.random.seed(0)



class ColTorchCore(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # Init config
        self.config = config

        # 3 bits per channel, 8 colors per channel, a total of 512 colors.
        self.num_symbols_per_channel = 2 ** 3
        self.num_symbols = self.num_symbols_per_channel ** 3
        self.gray_symbols = 256
        self.num_channels = 1

        self.enc_cfg = config.encoder
        self.dec_cfg = config.decoder
        self.hidden_size = self.config.get('hidden_size',
                                        self.dec_cfg.hidden_size)

        # stage can be 'encoder_decoder' or 'decoder'
        # 1. decoder -> loss only due to autoregressive model.
        # 2. encoder_decoder -> loss due to both the autoregressive and parallel
        # model.
        # encoder_only and all
        self.stage = config.get('stage', 'decoder')
        self.is_parallel_loss = 'encoder' in self.stage
        

