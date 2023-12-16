#!/bin/bash

# Create scaled directory and its subdirectories
mkdir -p scaled/images
mkdir -p scaled/pred

# Define models and their corresponding output directories
declare -A models
# models[hue_50_plus]='checkpoints/checkpoint_epoch50_hue.pth'
# models[sat_50_plus]='checkpoints/checkpoint_epoch33_saturation.pth'
# models[bright_50_plus]='checkpoints/checkpoint_epoch50_brightness.pth'
# models[contr_50_plus]='checkpoints/checkpoint_epoch50_contrast.pth'
# models[orig_50_plus]='checkpoints/checkpoint_epoch50_original.pth'
# models[orig_15_plus]='weights/original_15++.pth'
models[orig_75_32_-6]='checkpoints/checkpoint_epoch25_original.pth'

val='val++'

# Loop through all images in the val_aug/images directory
for image_path in data/${val}/images/*.png
do
    # Extract the basename of the image file without the extension
    image=$(basename "$image_path" .png)
    echo "Processing Image $image"

    # Upscale the image
    python utils/scaling.py "data/${val}/images/${image}.png" "scaled/images/${image}_upscaled.png" -s 2

    for model_key in "${!models[@]}"
    do
        model=${models[$model_key]}
        echo "Processing Model $model_key for Image $image"

        # Apply prediction model
        echo "Predicting Image: $image (Model: $model_key)"
        python predict.py -i "scaled/images/${image}_upscaled.png" -o "scaled/pred/mask_${image}_${model_key}.png" --model $model

        # Downscale the image
        echo "Downscaling Image: $image (Model: $model_key)"
        mkdir -p data/${val}/pred/$model_key
        python utils/scaling.py "scaled/pred/mask_${image}_${model_key}.png" "data/${val}/pred/$model_key/${image}.png" -s 0.5
    done
done

# Optionally, you can remove the scaled directory at the end
rm -r scaled

echo "All images processed successfully."
