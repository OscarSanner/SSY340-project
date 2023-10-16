import torch
import glob

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from PIL import Image
import numpy as np

from .models import MainModel
from .dataset import make_dataloaders
from .utils import visualize, lab_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

def ICT_colorize(source_folder, destination_folder, weights_path, use_gpu=False):
    print("Making dataloaders...")
    dl = make_dataloaders(source_folder=source_folder)

    def build_res_unet(n_input=1, n_output=2, size=256):
        body = create_body(resnet18(pretrained=True), n_in=n_input, cut=-2)
        #body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
        net_G = DynamicUnet(body, n_output, (size, size)).to(device)
        return net_G

    print("Building resnet...")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    # TODO: We might want ot remove this later 
    print("Loading resnet...")
    # net_G.load_state_dict(torch.load(f"{weights_path}/res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    print("Loading final model...")
    model.load_state_dict(torch.load(f"{weights_path}/final_model_weights.pt", map_location=device))

    model.net_G.eval()
    model.eval()
    
    print("Starting eval...")
    with torch.no_grad():
        for x, file_names in dl:
            output = model.forward_pred(x)
            L = model.L
            converted_output = lab_to_rgb(L, output)
            images_with_names = zip(converted_output, file_names)
            pil_images = [(Image.fromarray((image * 255).astype(np.uint8)), name) for image, name in images_with_names]
            for image, name in pil_images:
                image.save(f"{destination_folder}/{name}")
            
    
    #data = next(iter(dl))
    #visualize(model, data, save=True)
