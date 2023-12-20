#!/bin/bash

# Define models and their corresponding output directories
declare -A models
models[original]='weights/weights_original.pth'
models[hue]='weights/weights_hue.pth'
models[contrast]='weights/weights_contrast.pth'
models[brightness]='weights/weights_brightness.pth'
models[saturation]='weights/weights_saturation.pth'
# models[test]='weights/best/checkpoint_epoch150_hue.pth'

val='val++'

dbscan=500

up_scaling=2
down_scaling=0.5

# Create scaled directory and its subdirectories
mkdir -p scaled/images
mkdir -p scaled/pred

# Loop through all images in the val_aug/images directory
for image_path in data/${val}/images/*.png
do
    # Extract the basename of the image file without the extension
    image=$(basename "$image_path" .png)
    echo "Processing Image $image"

    # Upscale the image
    python utils/scaling.py "data/${val}/images/${image}.png" "scaled/images/${image}_upscaled.png" -s $up_scaling

    for model_key in "${!models[@]}"
    do
        model=${models[$model_key]}
        echo "  - Predicting for Model $model_key"

        # Apply prediction model
        # echo "Predicting Image: $image (Model: $model_key)"
        python predict.py -i "scaled/images/${image}_upscaled.png" -o "scaled/pred/mask_${image}_${model_key}.png" --model $model

        # Downscale the image
        # echo "Downscaling Image: $image (Model: $model_key)"
        mkdir -p data/${val}/pred/$model_key
        python utils/scaling.py "scaled/pred/mask_${image}_${model_key}.png" "data/${val}/pred/$model_key/${image}.png" -s $down_scaling
    done
done

# Remove the scaled directory at the end
rm -r scaled

echo "All predictions were generated successfully."

for model_key in "${!models[@]}"
do
    model=${models[$model_key]}
    echo "Evaluating Model $model_key"

    python evaluate.py $val $model_key

    echo "Performing DBSCAN with threshold $dbscan"

    python utils/DBSCAN_cleaning.py "./data/${val}/pred/${model_key}" "${model_key}_dbscan${dbscan}" --threshold $dbscan --type 'val'

    echo "Evaluating Model $model_key (with DBSCAN $dbscan)"

    python evaluate.py $val "${model_key}_dbscan${dbscan}"
done
