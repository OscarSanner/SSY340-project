from PIL import Image
from colorization.colorizers.eccv16_siggraph.execute import eccv16_colorize, siggraph17_colorize
from colorization.colorizers.ICT.execute import ICT_colorize 

import os
import datetime
import subprocess
import sys

raw_path = "./dataset/raw_color_data"
bw_path = "./dataset/bw_data"
gt_path = "./dataset/ground_truth"
pred_path = "./dataset/pred_data"
target_size = (256, 256)

# 1: bw pictures -> bw_data

def log(message):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

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

    if sys.platform.startswith('win'):
        print("System is Windows")
        venv_activate_command = ["powershell.exe", "-Command", "./venv/Scripts/Activate.ps1", ";"]
        move_final_colorized_images_command = ["powershell.exe", "-Command", "move", coltran_source_dir, coltran_dest_dir]
        remove_out_command = ["powershell.exe", "-Command", f"Remove-Item -Recurse -Force 'colorization/colorizers/coltran/out/'"]

    else:
        print("System is UNIX")
        venv_activate_command = ["venv/bin/activate", "&&"]
        move_final_colorized_images_command = ["mv", coltran_source_dir, coltran_dest_dir]
        remove_out_command = ["rm", "-r", "colorization/colorizers/coltran/out/"]
    
    colorize_command_step_1 = venv_activate_command + [
        "python", "-m", "colorization.colorizers.coltran.custom_colorize",
        "--config=colorization/colorizers/coltran/configs/colorizer.py",
        "--logdir=colorization/colorizers/coltran/weights/colorizer",
        f"--img_dir={source_path}",
        "--store_dir=colorization/colorizers/coltran/out/",
        "--mode=colorize"
    ]

    color_upsample_command_step_2 = venv_activate_command + [
        "python", "-m", "colorization.colorizers.coltran.custom_colorize",
        "--config=colorization/colorizers/coltran/configs/color_upsampler.py",
        "--logdir=colorization/colorizers/coltran/weights/color_upsampler",
        f"--img_dir={source_path}",
        "--store_dir=colorization/colorizers/coltran/out/",
        "--gen_data_dir=colorization/colorizers/coltran/out/stage1",
        "--mode=colorize"
    ]

    spatial_upsample_command_step_3 = venv_activate_command + [
        "python", "-m", "colorization.colorizers.coltran.custom_colorize",
        "--config=colorization/colorizers/coltran/configs/spatial_upsampler.py",
        "--logdir=colorization/colorizers/coltran/weights/spatial_upsampler",
        f"--img_dir={source_path}",
        "--store_dir=colorization/colorizers/coltran/out/",
        "--gen_data_dir=colorization/colorizers/coltran/out/stage2",
        "--mode=colorize"
    ]
    
    # Exectution
    log("Colorizing using: coltran")

    log("Step 1: Colorizer")
    subprocess.run(colorize_command_step_1, check=True)

    log("Step 2: Color upsampler")
    subprocess.run(color_upsample_command_step_2, check=True)
    
    log("Step 3: Spatial upsampler")
    subprocess.run(spatial_upsample_command_step_3, check=True)

    log("Step 4: Moving final colorized images")
    subprocess.run(move_final_colorized_images_command, shell=True, check=True)

    if cleanup_files:
        log("Step 5: Removing files in out/...")
        subprocess.run(remove_out_command, check=True)

if __name__ == "__main__":
    # ICT_colorize(bw_path, f"{pred_path}/ICT", "colorization/colorizers/ICT")
    process_eccv16_colorize(bw_path)
    process_siggraph17_colorize(bw_path)
    process_ICT_colorize(bw_path)
    process_coltran_colorize(bw_path, True)