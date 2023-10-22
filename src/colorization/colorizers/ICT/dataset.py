import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
import glob
import os
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

SIZE = 256

class ColorizationDataset(Dataset):
    def __init__(self, source_folder):
        self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)
        self.size = SIZE
        self.paths = glob.glob(f"{source_folder}/*.jpg")

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1 
        return {'L': L, 'ab': ab}, os.path.basename(self.paths[idx])

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=8, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
