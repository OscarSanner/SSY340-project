import glob
from PIL import Image
from colorization.models import EnsembleHeadColorizer, MeanChannelColorizer
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from torchvision import transforms
import torch
import os
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from piq import ssim, psnr

def retrieve_image(img_path):
    img = Image.open(img_path)
    rgb_img = img.convert("RGB")
    npa_img = np.array(rgb_img)
    return torch.Tensor(npa_img)

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

def calculate_psnr_ssim(gt_paths, coltran_path, ICT_path, eccv16_path, siggraph_path):
    avg_model = MeanChannelColorizer()
    ens_model = EnsembleHeadColorizer()
    
    checkpoint = torch.load('model_2023-Oct-19_22h04m05s.ckpt')
    ens_model.load_state_dict(checkpoint['model_state_dict'])
    ens_model.eval()

    def retrieve_images_in_batches(paths, amount):
        BATCH_SIZE = 100
        all_imgs = []
        for i in range(0, min(amount, len(paths)), BATCH_SIZE):
            batch_paths = paths[i:i+BATCH_SIZE]
            imgs = torch.stack([retrieve_image(p) for p in batch_paths])
            all_imgs.append(imgs)
        return torch.cat(all_imgs, dim=0)

    print("Retrieving images")
    gt_imgs = retrieve_images_in_batches(gt_paths, 256).permute(0,3,1,2)
    print("Retrieving images")
    coltran = retrieve_images_in_batches(coltran_path, 256).permute(0,3,1,2)
    print("Retrieving images")
    ICT_imgs = retrieve_images_in_batches(ICT_path, 256).permute(0,3,1,2)
    print("Retrieving images")
    eccv_imgs = retrieve_images_in_batches(eccv16_path, 256).permute(0,3,1,2)
    print("Retrieving images")
    siggraph_imgs = retrieve_images_in_batches(siggraph_path, 256).permute(0,3,1,2)

    combined_imgs = torch.cat([eccv_imgs, siggraph_imgs, ICT_imgs, coltran], dim=1)
    averaged_imgs = torch.stack([avg_model.forward(img) for img in combined_imgs])

    averaged_imgs[0]

    def tensor_to_PIL(tensor):
        # Move tensor to CPU and detach if necessary
        tensor = tensor.cpu().detach()
        # Convert tensor to [H, W, C] format
        numpy_img = tensor.permute(1, 2, 0).numpy()
        # Convert numpy array to PIL Image
        return Image.fromarray((numpy_img * 255).astype(np.uint8))
    
    predicted_imgs = []
    for i in range(len(coltran_path)):
        eccv16_tensor = image_to_tensor(eccv16_path[i])
        siggraph_tensor = image_to_tensor(siggraph_path[i])
        ICT_tensor = image_to_tensor(ICT_path[i])
        coltran_tensor = image_to_tensor(coltran_path[i])
        input_data = torch.cat((eccv16_tensor, siggraph_tensor, ICT_tensor, coltran_tensor), axis=0)

        ens_model.eval()
        with torch.no_grad(): 
            prediction = ens_model(input_data.unsqueeze(0)).squeeze(0)
        
        open = tensor_to_image(prediction)
        pil_img = tensor_to_PIL(averaged_imgs[i])

        filename = os.path.basename(siggraph_path[i])
        open.save(f"dataset/pred_data/ens/{filename}")
        pil_img.save(f"dataset/pred_data/avg/{filename}")
        

        prediction = prediction.permute(1, 2, 0)

        # Convert LAB to RGB
        rgb_image = lab2rgb(prediction.cpu().numpy())

        # Convert the resulting numpy array back to a PyTorch tensor
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
        rgb_tensor = rgb_tensor * 255


        predicted_imgs.append(rgb_tensor)

    predicted_tensors = torch.stack(predicted_imgs)

    print("Calculating metrics")
    
    predicted_psnr_score = psnr(predicted_tensors, gt_imgs, reduction='mean', data_range=255), 
    predicted_ssim_score = ssim(predicted_tensors, gt_imgs, reduction='mean', data_range=255),

    combined_psnr_score = psnr(averaged_imgs, gt_imgs, reduction='mean', data_range=255), 
    combined_ssim_score = ssim(averaged_imgs, gt_imgs, reduction='mean', data_range=255),

    coltran_psnr_score = psnr(coltran, gt_imgs, reduction='mean', data_range=255), 
    coltran_ssim_score = ssim(coltran, gt_imgs, reduction='mean', data_range=255),

    ict_psnr_score = psnr(ICT_imgs, gt_imgs, reduction='mean', data_range=255), 
    ict_ssim_score = ssim(ICT_imgs, gt_imgs, reduction='mean', data_range=255),

    eccv_psnr_score = psnr(eccv_imgs, gt_imgs, reduction='mean', data_range=255), 
    eccv_ssim_score = ssim(eccv_imgs, gt_imgs, reduction='mean', data_range=255),

    siggraph_psnr_score = psnr(siggraph_imgs, gt_imgs, reduction='mean', data_range=255), 
    siggraph_ssim_score = ssim(siggraph_imgs, gt_imgs, reduction='mean', data_range=255),

    print(f"Predicted PSNR: {predicted_psnr_score}")
    print(f"Predicted SSIM: {predicted_ssim_score}")
    print("---")
    print(f"Combined PSNR: {combined_psnr_score}")
    print(f"Combined SSIM: {combined_ssim_score}")
    print("---")
    print(f"Coltran PSNR: {coltran_psnr_score}")
    print(f"Coltran SSIM: {coltran_ssim_score}")
    print("---")
    print(f"ICT PSNR: {ict_psnr_score}")
    print(f"ICT SSIM: {ict_ssim_score}")
    print("---")
    print(f"Eccv PSNR: {eccv_psnr_score}")
    print(f"Eccv SSIM: {eccv_ssim_score}")
    print("---")
    print(f"Siggraph PSNR: {siggraph_psnr_score}")
    print(f"Siggraph SSIM: {siggraph_ssim_score}")
    

    return (combined_psnr_score, combined_ssim_score, coltran_psnr_score, 
            coltran_ssim_score, ict_psnr_score, ict_ssim_score, 
            eccv_psnr_score, eccv_ssim_score, siggraph_psnr_score, 
            siggraph_ssim_score, predicted_ssim_score ,predicted_psnr_score)

def main():
    
    plot_psnr_ssim_bars()

def plot_psnr_ssim_bars():

    ground_truth_image_path = glob.glob(f"dataset/ground_truth/*")
    coltran_image_path = glob.glob(f"dataset/pred_data/coltran/*")
    ICT_image_path = glob.glob(f"dataset/pred_data/ICT/*")
    eccv16_image_path = glob.glob(f"dataset/pred_data/eccv16/*")
    siggraph_image_path = glob.glob(f"dataset/pred_data/siggraph/*")

    assert len(ground_truth_image_path) == len(coltran_image_path) == len(ICT_image_path) == len(eccv16_image_path) == len(siggraph_image_path)

    ground_truth_image_path.sort()
    coltran_image_path.sort()
    ICT_image_path.sort()
    eccv16_image_path.sort()
    siggraph_image_path.sort()
    (combined_psnr_score, combined_ssim_score, coltran_psnr_score, 
     coltran_ssim_score, ict_psnr_score, ict_ssim_score, 
     eccv_psnr_score, eccv_ssim_score, siggraph_psnr_score, 
     siggraph_ssim_score, predicted_ssim_score, predicted_psnr_score) = calculate_psnr_ssim(ground_truth_image_path, coltran_image_path, ICT_image_path, eccv16_image_path, siggraph_image_path)


    psnr_scores = [
        coltran_psnr_score[0].item(), ict_psnr_score[0].item(), eccv_psnr_score[0].item(),
        siggraph_psnr_score[0].item(), combined_psnr_score[0].item(), predicted_psnr_score[0].item()
    ]

    ssim_scores = [
        coltran_ssim_score[0].item(), ict_ssim_score[0].item(), eccv_ssim_score[0].item(),
        siggraph_ssim_score[0].item(), combined_ssim_score[0].item(), predicted_ssim_score[0].item()
    ]

    colors = ['C0', 'C0', 'C0', 'C0', 'green', 'orange']
    labels = ['Coltran', 'U-Net/GAN', 'ECCV', 'Siggraph', 'Averaged', 'Ensamble']

    # Plotting the PSNR scores
    plt.figure(figsize=(6, 6))
    plt.bar(labels, psnr_scores, color=colors, width=0.9)
    plt.xticks(rotation=45)  # Tilt text under bars
    plt.title('PSNR Scores')
    plt.ylabel('PSNR Value')
    plt.xlabel('Models')
    plt.tight_layout()  # Adjust layout for saving
    plt.savefig('psnr_scores.png')  # Save the figure
    plt.show()

    # Plotting the SSIM scores
    plt.figure(figsize=(6, 6))
    plt.bar(labels, ssim_scores, color=colors, width=0.9)
    plt.xticks(rotation=45)  # Tilt text under bars
    plt.title('SSIM Scores')
    plt.ylabel('SSIM Value')
    plt.xlabel('Models')
    plt.tight_layout()  # Adjust layout for saving
    plt.savefig('ssim_scores.png')  # Save the figure
    plt.show()

    # Create combined plot with side-by-side subplots
    plt.figure(figsize=(10, 6))
    
    # PSNR subplot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.bar(labels, psnr_scores, color=colors, width=0.9)
    plt.xticks(rotation=45)  # Tilt text under bars
    plt.title('PSNR Scores')
    plt.ylabel('PSNR Value')
    plt.xlabel('Models')
    
    # SSIM subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.bar(labels, ssim_scores, color=colors, width=0.9)
    plt.xticks(rotation=45)  # Tilt text under bars
    plt.title('SSIM Scores')
    plt.ylabel('SSIM Value')
    plt.xlabel('Models')
    
    plt.tight_layout()
    plt.savefig('combined_scores.png')
    plt.show()

if __name__ == "__main__":
    main()

