import os
import shutil

bird_data_dir = "./bird_data/CUB_200_2011/images"
target_dir = "./dataset/raw_color_data"

def copy_files_in_range(start_index, end_index):
    """
    Copies files from bird_data_dir to target_dir based on given start and end indices,
    and prepends the file name with its index.
    
    Args:
    - start_index (int): Start index (inclusive).
    - end_index (int): End index (exclusive).
    """
    if not os.path.exists(bird_data_dir):
        raise Exception("Can't find directory: {bird_data_dir}")
    
    all_files = []
    for root, _, files in os.walk(bird_data_dir):
        for f in files:
            all_files.append(os.path.join(root, f))
    
    all_files.sort()
    files_to_copy = all_files[start_index:end_index]
    print(f"Moving {len(files_to_copy)} pictures from {bird_data_dir} to {target_dir}")

    for idx, file_path in enumerate(files_to_copy):
        new_file_name = f"{idx}-{os.path.basename(file_path)}"
        target_path = os.path.join(target_dir, new_file_name)
        
        shutil.copy(file_path, target_path)

copy_files_in_range(57, 62)
