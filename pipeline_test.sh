#!/bin/bash

# Define models and their corresponding output directories
declare -A models
models[original]='weights/weights_original.pth'
models[hue]='weights/weights_hue.pth'
models[contrast]='weights/weights_contrast.pth'
models[brightness]='weights/weights_brightness.pth'
models[saturation]='weights/weights_saturation.pth'
# models[test]='weights/best/checkpoint_epoch150_hue.pth'

dbscan=500

up_scaling=2
down_scaling=0.5

# Create scaled directory and its subdirectories
mkdir -p scaled/images
mkdir -p scaled/pred

# Start and end indices for image iteration
start_image_i=1
end_image_i=50

for image_index in $(seq $start_image_i $end_image_i)
do
    echo "Processing Image $image_index"

    # Upscale the image
    python utils/scaling.py "data/test_set_images/test_${image_index}/test_${image_index}.png" "scaled/images/${image_index}_upscaled.png" -s $up_scaling

    for model_key in "${!models[@]}"
    do
        model=${models[$model_key]}
        echo "  - Predicting for Model $model_key"

        # Apply prediction model
        # echo "Predicting Image $image_index (Model: $model_key)"
        python predict.py -i "scaled/images/${image_index}_upscaled.png" -o "scaled/pred/mask_${image_index}_${model_key}.png" --model $model

        # Downscale the image
        # echo "Downscaling Image $image_index (Model: $model_key)"
        mkdir -p predictions/$model_key
        python utils/scaling.py "scaled/pred/mask_${image_index}_${model_key}.png" "predictions/$model_key/pred_test_${image_index}.png" -s $down_scaling
    done
done

# Remove the scaled directory at the end
rm -r scaled

echo "All predictions were generated successfully."

for model_key in "${!models[@]}"
do
    model=${models[$model_key]}
    echo "Evaluating Model $model_key"

    echo "Performing DBSCAN with threshold $dbscan"

    python utils/DBSCAN_cleaning.py "./predictions/${model_key}" "${model_key}_dbscan${dbscan}" --threshold $dbscan --type 'test'

done
