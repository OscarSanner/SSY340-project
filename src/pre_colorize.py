from PIL import Image
from colorization.colorizers.eccv16_siggraph.execute import eccv16_colorize, siggraph17_colorize
from colorization.colorizers.ICT.execute import ICT_colorize 

import os
import datetime
import subprocess
import sys

from tensorflow.keras import backend as K
import torch

import gdown

raw_path = "./dataset/raw_color_data"
bw_path = "./dataset/bw_data"
gt_path = "./dataset/ground_truth"
pred_path = "./dataset/pred_data"
target_size = (256, 256)

def log(message):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

def make_image_dirs():
    log("Creating directories")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(bw_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(f"{pred_path}/coltran", exist_ok=True)
    os.makedirs(f"{pred_path}/ICT", exist_ok=True)
    os.makedirs(f"{pred_path}/siggraph", exist_ok=True)
    os.makedirs(f"{pred_path}/eccv16", exist_ok=True)

def prep_resized_bw_data(source_path):
    for filename in os.listdir(source_path):
        f = os.path.join(source_path, filename)
        img = Image.open(f)
        img = img.resize(target_size)
        img.save(f"{gt_path}/{filename}")
        img = img.convert("L")
        img.save(f"{bw_path}/{filename}")

def process_eccv16_colorize(source_path):
    log("Colorizing using: eccv16")
    for filename in os.listdir(source_path):
        eccv16_colorize(source_path, f"{pred_path}/eccv16/", filename)
    log("eccv16 complete!")
        
def process_siggraph17_colorize(source_path):
    log("Colorizing using: siggraph17")
    for filename in os.listdir(source_path):
        siggraph17_colorize(source_path, f"{pred_path}/siggraph/", filename)
    log("siggraph17 complete!")
        
def process_ICT_colorize(source_path):
    log("Colorizing using: ICT")
    ICT_colorize(source_path, f"{pred_path}/ICT", "colorization/colorizers/ICT")
    log("ICT complete!")

def process_coltran_colorize(source_path, cleanup_files=False):
    coltran_source_dir = "colorization/colorizers/coltran/out/final/*.jpg"
    coltran_dest_dir = f"{pred_path}/coltran"

    colorize_command_step_1 = f"./pre_colorize_helpers.sh coltran_step_1 {source_path}"
    color_upsample_command_step_2 = f"./pre_colorize_helpers.sh coltran_step_2 {source_path}"
    spatial_upsample_command_step_3 = f"./pre_colorize_helpers.sh coltran_step_3 {source_path}"

    move_final_colorized_images_command = f"mv {coltran_source_dir} {coltran_dest_dir}"
    remove_out_command = ["rm", "-r", "colorization/colorizers/coltran/out/"]

    # Exectution
    log("Colorizing using: coltran")
    
    log("Step 1: Colorizer")
    #subprocess.run(colorize_command_step_1, check=True, executable="/bin/bash", shell=True)
    K.clear_session()

    log("Step 2: Color upsampler")
    #subprocess.run(color_upsample_command_step_2, check=True, executable="/bin/bash", shell=True)
    K.clear_session()
    
    log("Step 3: Spatial upsampler")
    subprocess.run(spatial_upsample_command_step_3, check=True, executable="/bin/bash", shell=True)
    K.clear_session()

    log("Step 4: Moving final colorized images")
    subprocess.run(move_final_colorized_images_command, shell=True, check=True, executable="/bin/bash")

    if cleanup_files:
        log("Step 5: Removing files in out/...")
        subprocess.run(remove_out_command, check=True)


def download_from_gdrive():
    weights_ICT_path = "./colorization/colorizers/ICT/final_model_weights.pt"
    weights_ICT_link = "https://drive.google.com/uc?id=1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV"

    log("Downloading weights for ICT")
    if not os.path.exists(weights_ICT_path):
        gdown.download(weights_ICT_link, weights_ICT_path, quiet=False)
    else:
        print(f"Weights for ICT already exists at {weights_ICT_path}. Skipping download.")

    weights_coltran_path = "./colorization/colorizers/coltran/weights"
    coltran_intermediate_folder = "./coltran_tmp"

    coltran_fetch_command = ["wget", "https://storage.googleapis.com/gresearch/coltran/coltran.zip", "-P", coltran_intermediate_folder]
    coltran_unzip_command = ["unzip",  f"{coltran_intermediate_folder}/coltran.zip", "-d", coltran_intermediate_folder]
    coltran_chmod_command = ["chmod", "-R", "700", f"./{coltran_intermediate_folder}/coltran"]
    coltran_move_command = ["mv", f"{coltran_intermediate_folder}/coltran", weights_coltran_path]
    coltran_remove_temp_command = ["rm", "-frd", coltran_intermediate_folder]

    log("Downloading weights for coltran")
    if not os.path.exists(weights_coltran_path):
        subprocess.run(coltran_fetch_command, check=True)
        subprocess.run(coltran_unzip_command, check=True)
        subprocess.run(coltran_chmod_command, check=True)
        subprocess.run(coltran_move_command, check=True)
        subprocess.run(coltran_remove_temp_command, check=True)
    else:
        print(f"Weights for coltran already exists at {weights_coltran_path}. Skipping download.")
    log("Weights for eccv16 and siggraph are downloaded when the model is instantiated.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    make_image_dirs()
    download_from_gdrive()
    prep_resized_bw_data(raw_path)
    process_siggraph17_colorize(bw_path)
    process_eccv16_colorize(bw_path)
    process_ICT_colorize(bw_path)
    process_coltran_colorize(bw_path, True)