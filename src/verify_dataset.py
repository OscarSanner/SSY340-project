#!/usr/bin/env python3

import os

data_dir = "./dataset/"
pred_dir = "./dataset/pred_data/"
coltran_files = set()
eccv16_files = set()
siggraph_files = set()
ICT_files = set()

ground_truth_files = set()
bw_files = set()
raw_data_files = set()

for root, _, files in os.walk(f"{pred_dir}/coltran"):
    for f in files:
        coltran_files.add(f)

for root, _, files in os.walk(f"{pred_dir}/eccv16"):
    for f in files:
        eccv16_files.add(f)

for root, _, files in os.walk(f"{pred_dir}/siggraph"):
    for f in files:
        siggraph_files.add(f)

for root, _, files in os.walk(f"{pred_dir}/ICT"):
    for f in files:
        ICT_files.add(f)



for root, _, files in os.walk(f"{data_dir}/ground_truth"):
    for f in files:
        ground_truth_files.add(f)

for root, _, files in os.walk(f"{data_dir}/bw_data"):
    for f in files:
        bw_files.add(f)

for root, _, files in os.walk(f"{data_dir}/raw_color_data"):
    for f in files:
        raw_data_files.add(f)


set_pairs = {
    ('coltran_files', 'eccv16_files'): (coltran_files, eccv16_files),
    ('coltran_files', 'siggraph_files'): (coltran_files, siggraph_files),
    ('coltran_files', 'ICT_files'): (coltran_files, ICT_files),
    ('eccv16_files', 'siggraph_files'): (eccv16_files, siggraph_files),
    ('eccv16_files', 'ICT_files'): (eccv16_files, ICT_files),
    ('siggraph_files', 'ICT_files'): (siggraph_files, ICT_files),

    ('siggraph_files', 'bw_files'): (siggraph_files, bw_files),
    ('siggraph_files', 'raw_data_files'): (siggraph_files, raw_data_files),
    ('siggraph_files', 'ground_truth_files'): (siggraph_files, ground_truth_files),
}

for pair, sets in set_pairs.items():
    set1, set2 = sets
    diff1 = set1 - set2
    diff2 = set2 - set1

    print(f"Difference between {pair[0]} and {pair[1]}:")
    print(f"{pair[0]} - {pair[1]}: {diff1}")
    print(f"{pair[1]} - {pair[0]}: {diff2}")
    print("-------------------------------------------------")

