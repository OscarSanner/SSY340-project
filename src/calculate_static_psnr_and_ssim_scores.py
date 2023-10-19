import glob
from PIL import Image
from colorization.models import MeanChannelColorizer
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from torchvision import transforms
import torch
from piq import ssim, psnr

def retrieve_image(img_path):
    img = Image.open(img_path)
    rgb_img = img.convert("RGB")
    npa_img = np.array(rgb_img)
    return torch.Tensor(npa_img)

def calculate_psnr_ssim(gt_paths, coltran_path, ICT_path, eccv16_path, siggraph_path):
    model = MeanChannelColorizer()

    def retrieve_images_in_batches(paths, amount):
        BATCH_SIZE = 100
        all_imgs = []
        for i in range(0, min(amount, len(paths)), BATCH_SIZE):
            batch_paths = paths[i:i+BATCH_SIZE]
            imgs = torch.stack([retrieve_image(p) for p in batch_paths])
            all_imgs.append(imgs)
        return torch.cat(all_imgs, dim=0)


    gt_imgs = retrieve_images_in_batches(gt_paths, 256).permute(0,3,1,2)
    coltran = retrieve_images_in_batches(coltran_path, 256).permute(0,3,1,2)
    ICT_imgs = retrieve_images_in_batches(ICT_path, 256).permute(0,3,1,2)
    eccv_imgs = retrieve_images_in_batches(eccv16_path, 256).permute(0,3,1,2)
    siggraph_imgs = retrieve_images_in_batches(siggraph_path, 256).permute(0,3,1,2)

    combined_imgs = torch.cat([coltran, ICT_imgs, eccv_imgs, siggraph_imgs], dim=1)
    averaged_imgs = torch.stack([model.forward(img) for img in combined_imgs])

    combined_psnr_score = psnr(averaged_imgs, gt_imgs, reduction='mean', data_range=255), 
    combined_ssim_score = ssim(averaged_imgs, gt_imgs, reduction='mean', data_range=255),

    coltran_psnr_score = psnr(coltran, gt_imgs, reduction='mean', data_range=255), 
    coltran_ssim_score = ssim(coltran, gt_imgs, reduction='mean', data_range=255),

    ict_psnr_score = psnr(ICT_imgs, gt_imgs, reduction='mean', data_range=255), 
    ict_ssim_score = ssim(ICT_imgs, gt_imgs, reduction='mean', data_range=255),

    eccv_psnr_score = psnr(eccv_imgs, gt_imgs, reduction='mean', data_range=255), 
    eccv_ssim_score = ssim(eccv_imgs, gt_imgs, reduction='mean', data_range=255),

    siggraph_psnr_score = psnr(gt_imgs, gt_imgs, reduction='mean', data_range=255), 
    siggraph_ssim_score = ssim(gt_imgs, gt_imgs, reduction='mean', data_range=255),

    """
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
    print("---")
    """

    return (combined_psnr_score, combined_ssim_score, coltran_psnr_score, 
            coltran_ssim_score, ict_psnr_score, ict_ssim_score, 
            eccv_psnr_score, eccv_ssim_score, siggraph_psnr_score, 
            siggraph_ssim_score)

def main():
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
    
    calculate_psnr_ssim(ground_truth_image_path, coltran_image_path, ICT_image_path, eccv16_image_path, siggraph_image_path)

if __name__ == "__main__":
    main()
