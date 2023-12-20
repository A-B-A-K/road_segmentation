#!/bin/bash

# Create scaled directory and its subdirectories
mkdir -p scaled/images
mkdir -p scaled/pred

# Start and end indices for image iteration
start_image_i=1
end_image_i=50

# Define models and their corresponding output directories
declare -A models
# models[hue_50_plus]='checkpoints/checkpoint_epoch50_hue.pth'
# models[sat_50_plus]='checkpoints/checkpoint_epoch45_saturation.pth'
# models[bright_50_plus]='checkpoints/checkpoint_epoch50_brightness.pth'
# models[contr_50_plus]='checkpoints/checkpoint_epoch50_contrast.pth'
# models[orig_50_plus]='checkpoints/checkpoint_epoch50_original.pth'
# models[orig_best_hun]='weights/models++/checkpoint_epoch100_original.pth'
# models[hue_best_bs]='checkpoints/checkpoint_epoch150_hue.pth'
models[sat_best_bs]='weights/best/checkpoint_epoch150_saturation.pth'

for image_index in $(seq $start_image_i $end_image_i)
do
    echo "Processing Image $image_index"

    # Upscale the image
    python utils/scaling.py "data/test_set_images/test_${image_index}/test_${image_index}.png" "scaled/images/${image_index}_upscaled.png" -s 2

    for model_key in "${!models[@]}"
    do
        model=${models[$model_key]}
        echo "Processing Model $model_key for Image $image_index"

        # Apply prediction model
        echo "Predicting Image $image_index (Model: $model_key)"
        python predict.py -i "scaled/images/${image_index}_upscaled.png" -o "scaled/pred/mask_${image_index}_${model_key}.png" --model $model

        # Downscale the image
        echo "Downscaling Image $image_index (Model: $model_key)"
        mkdir -p predictions/$model_key
        python utils/scaling.py "scaled/pred/mask_${image_index}_${model_key}.png" "predictions/$model_key/pred_test_${image_index}.png" -s 0.5
    done
done

# Optionally, you can remove the scaled directory at the end
rm -r scaled

echo "All images processed successfully."
