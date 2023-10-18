import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
import glob
import os
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from itertools import chain
from PIL import Image

SIZE = 256

class ColorizationDataset(Dataset):
    def __init__(self, dataset_folder):
        self.transforms = transforms.Resize((SIZE, SIZE))
        # verify_dataset()
        self.image_names = [path.split("/")[-1] for path in glob.glob(f"{dataset_folder}/ground_truth/*.jpg")]
        self.image_names.sort()
        
        self.coltran_path = f"{dataset_folder}/pred_data/coltran"
        self.ICT_path = f"{dataset_folder}/pred_data/ICT"
        self.eccv16_path = f"{dataset_folder}/pred_data/eccv16"
        self.siggraph_path = f"{dataset_folder}/pred_data/siggraph"
        self.ground_truth_path = f"{dataset_folder}/ground_truth"

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        
        eccv16_tensor = self._retrieve_image(f"{self.eccv16_path}/{file_name}")
        siggraph_tensor = self._retrieve_image(f"{self.siggraph_path}/{file_name}")
        ICT_tensor = self._retrieve_image(f"{self.ICT_path}/{file_name}")
        coltran_tensor = self._retrieve_image(f"{self.coltran_path}/{file_name}")

        input_data = torch.cat((eccv16_tensor, siggraph_tensor, ICT_tensor, coltran_tensor), axis=0)

        true_data = self._retrieve_image(f"{self.ground_truth_path}/{file_name}")

        # Shape of input_data: torch.Size([12, 256, 256])
        # Shape of true_data: torch.Size([3, 256, 256])

        return input_data, true_data

    def __len__(self):
        return len(self.image_names)

    def _retrieve_image(self, img_path):
        img = Image.open(img_path)
        rgb_img = img.convert("RGB")
        npa_img = np.array(rgb_img)
        lab_img = rgb2lab(npa_img).astype("float32")
        return transforms.ToTensor()(lab_img)
