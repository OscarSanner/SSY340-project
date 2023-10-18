import os
pred_dir = "./dataset/pred_data/"
coltran_files = set()
correct_names = {}

for root, _, files in os.walk(f"{pred_dir}/eccv16"):
    for f in files:
        idx, name = f.split("-")
        correct_names[name] = idx


for root, _, files in os.walk(f"{pred_dir}/coltran"):
    for f in files:
        _, name = f.split("-")

        if name not in correct_names:
            raise Exception("File {name} in coltran was not found in eccv16!")
            
        new_idx = correct_names[name]
        new_name = f"{new_idx}-{name}"
        os.rename(f"{pred_dir}/coltran/{f}", f"{pred_dir}/coltran/{new_name}")
