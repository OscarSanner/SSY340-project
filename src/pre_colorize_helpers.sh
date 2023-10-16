#!/bin/bash

coltran_step_1() {
    source_path="$1"
    python3 -m colorization.colorizers.coltran.custom_colorize \
        --config=colorization/colorizers/coltran/configs/colorizer.py \
        --logdir=colorization/colorizers/coltran/weights/colorizer \
        --img_dir="$source_path" \
        --store_dir=colorization/colorizers/coltran/out/ \
        --mode=colorize
}

# Function for Step 2: Color Upsampler
coltran_step_2() {
    source_path="$1"

    python3 -m colorization.colorizers.coltran.custom_colorize \
        --config=colorization/colorizers/coltran/configs/color_upsampler.py \
        --logdir=colorization/colorizers/coltran/weights/color_upsampler \
        --img_dir="$source_path" \
        --store_dir=colorization/colorizers/coltran/out/ \
        --gen_data_dir=colorization/colorizers/coltran/out/stage1 \
        --mode=colorize
}

# Function for Step 3: Spatial Upsampler
coltran_step_3() {
    source_path="$1"

    python3 -m colorization.colorizers.coltran.custom_colorize \
        --config=colorization/colorizers/coltran/configs/spatial_upsampler.py \
        --logdir=colorization/colorizers/coltran/weights/spatial_upsampler \
        --img_dir="$source_path" \
        --store_dir=colorization/colorizers/coltran/out/ \
        --gen_data_dir=colorization/colorizers/coltran/out/stage2 \
        --mode=colorize
}

case "$1" in
    "coltran_step_1")
        coltran_step_1 "$2"
        ;;
    "coltran_step_2")
        coltran_step_2 "$2"
        ;;
    "coltran_step_3")
        coltran_step_3 "$2"
        ;;
    *)
        echo "Invalid option: $1"
        exit 1
        ;;
esac