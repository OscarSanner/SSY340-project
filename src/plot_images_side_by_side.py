import argparse
import os
import glob
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from colorization.models import EnsembleHeadColorizer, MeanChannelColorizer
from piq import ssim, psnr

def main(image_index, ckpt_path):
    current_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_path)

    if not os.path.basename(current_directory) == "src":
        raise ValueError("The script needs to be ran from 'src' directory to work")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"The provided checkpoint file '{ckpt_path}' does not exist.")

    ground_truth_image_path = glob.glob(f"dataset/ground_truth/{image_index}-*")[0]
    bw_image_path = glob.glob(f"dataset/bw_data/{image_index}-*")[0]
    coltran_image_path = glob.glob(f"dataset/pred_data/coltran/{image_index}-*")[0]
    ICT_image_path = glob.glob(f"dataset/pred_data/ICT/{image_index}-*")[0]
    eccv16_image_path = glob.glob(f"dataset/pred_data/eccv16/{image_index}-*")[0]
    siggraph_image_path = glob.glob(f"dataset/pred_data/siggraph/{image_index}-*")[0]

    checkpoint = torch.load(ckpt_path)
    saved_model = EnsembleHeadColorizer()
    saved_model.load_state_dict(checkpoint['model_state_dict'])

    eccv16_tensor = image_to_tensor(eccv16_image_path)
    siggraph_tensor = image_to_tensor(siggraph_image_path)
    ICT_tensor = image_to_tensor(ICT_image_path)
    coltran_tensor = image_to_tensor(coltran_image_path)
    input_data = torch.cat((eccv16_tensor, siggraph_tensor, ICT_tensor, coltran_tensor), axis=0)

    saved_model.eval()
    with torch.no_grad(): 
        mean_channel_colorizer = MeanChannelColorizer()
        avg = mean_channel_colorizer.forward(input_data)
        prediction = saved_model.forward(input_data)
    
    predicted_colorization = tensor_to_image(prediction)
    avg_colorization = tensor_to_image(avg)
    ground_truth_colorization = Image.open(ground_truth_image_path)
    bw_colorization = Image.open(bw_image_path)
    coltran_colorization = Image.open(coltran_image_path)
    ICT_colorization = Image.open(ICT_image_path)
    eccv16_colorization = Image.open(eccv16_image_path)
    siggraph_colorization = Image.open(siggraph_image_path)

    # Setting up the images and titles
    images = [
        eccv16_colorization, 
        siggraph_colorization,
        ICT_colorization, 
        coltran_colorization,
        bw_colorization,
        ground_truth_colorization,
        avg_colorization,
        predicted_colorization
    ]

    # Create a figure with 2x4 subplots
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Titles for each subplot
    titles = [
        "ECCV16", "Siggraph", "U-Net/GAN", "Coltran",
        "Grayscale", "Ground Truth", "Average", "Ensamble"
    ]

    # Display each image in a subplot
    for ax, img, title in zip(axes, images, titles):
        if title == "Grayscale":
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # Show the entire figure
    plt.tight_layout()
    plt.show()


# Example:
# python run_model.py -i 1756 -c model_2023-Oct-18_21h49m11s.ckpt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image based on its index.")
    parser.add_argument("-i", "--image_index", type=int, required=True, help="Index of the image to colorize")
    parser.add_argument("-c", "--ckpt_path", type=str, required=True, help="Path to the model (.ckpt) file.")

    args = parser.parse_args()
    main(args.image_index, args.ckpt_path)