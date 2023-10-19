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

def image_to_tensor(img_path):
    img = Image.open(img_path)
    rgb_img = img.convert("RGB")
    npa_img = np.array(rgb_img)
    lab_img = rgb2lab(npa_img).astype("float32")
    return transforms.ToTensor()(lab_img)

def tensor_to_image(tensor):
    npa_img = tensor.cpu().numpy().transpose((1, 2, 0))
    rgb_img = lab2rgb(npa_img).clip(0, 1)
    img = Image.fromarray((rgb_img * 255).astype(np.uint8), 'RGB')
    return img

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
        prediction = saved_model(input_data.unsqueeze(0)).squeeze(0)
    
    mean_channel_colorizer = MeanChannelColorizer()
    mean_predicted_image = mean_channel_colorizer.forward(input_data)
    
    predicted_colorization = tensor_to_image(prediction)
    ground_truth_colorization = Image.open(ground_truth_image_path)
    bw_colorization = Image.open(bw_image_path)
    coltran_colorization = Image.open(coltran_image_path)
    ICT_colorization = Image.open(ICT_image_path)
    eccv16_colorization = Image.open(eccv16_image_path)
    siggraph_colorization = Image.open(siggraph_image_path)

    # Setting up the images and titles
    images = [
        bw_colorization,
        coltran_colorization, 
        ICT_colorization, 
        eccv16_colorization, 
        siggraph_colorization,
        predicted_colorization,
        ground_truth_colorization
    ]

    titles = ["BW", "Coltran", "ICT", "ECCV16", "SIGGRAPH", "Predicted", "Ground Truth"]

    # Create a figure with custom size
    fig, axs = plt.subplots(3, 5, figsize=(20, 15))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing between subplots

    # Hide all axes at first
    for ax in axs.ravel():
        ax.axis('off')

    # Plot BW image on the top center
    axs[0, 2].imshow(images[0], cmap='gray')
    axs[0, 2].set_title(titles[0])

    # Plot the four images below the BW image
    for i in range(1, 5):
        axs[1, i-1].imshow(images[i])
        axs[1, i-1].set_title(titles[i])
        axs[1, i-1].axis('on')


    # Plot the predicted_colorization in the center of the third row
    axs[2, 2].imshow(images[5])
    axs[2, 2].set_title(titles[5])

    # Plot the ground_truth_colorization next to the predicted image
    axs[2, 3].imshow(images[6])
    axs[2, 3].set_title(titles[6])

    plt.tight_layout()

    image_name = ground_truth_image_path.split("/")[-1]
    plt.savefig(f"plot_colorization_{image_name}.png")


# Example:
# python run_model.py -i 1756 -c model_2023-Oct-18_21h49m11s.ckpt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image based on its index.")
    parser.add_argument("-i", "--image_index", type=int, required=True, help="Index of the image to colorize")
    parser.add_argument("-c", "--ckpt_path", type=str, required=True, help="Path to the model (.ckpt) file.")

    args = parser.parse_args()
    main(args.image_index, args.ckpt_path)