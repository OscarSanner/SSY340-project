from PIL import Image
from colorization.colorizers.eccv16_siggraph.execute import eccv16_colorize, siggraph17_colorize
import os
import datetime

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

# 2: eccv16 -> pred/eccv16
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


# 2: siggraph -> pred/siggraph
# 2: ICT -> pred/ICT
# 2: coltran -> pred/coltran

process_eccv16_colorize(bw_path)
process_siggraph17_colorize(bw_path)
