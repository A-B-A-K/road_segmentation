#!/bin/bash

validation_dir="data/val/images"
# Start and end indices for image and tile iteration
order=1
start_tile_i=1
end_tile_i=$((order * order))
model=checkpoints/checkpoint_epoch15.pth
predict_output="validation_e15"

for image_file in "${validation_dir}"/*.png; do
    # Extract the base name of the image (without extension)
    image=$(basename "${image_file}" .png)

    echo "Processing Image ${image}"
    python utils/tile_creator.py "data/val/images/${image}.png" "tiles/tiles_${image}" --order $order


    # Loop through the tile indices
    for tile_index in $(seq $start_tile_i $end_tile_i); do
        echo "Predicting Tile ${tile_index} for Image ${image}"
        python predict.py -i "tiles/tiles_${image}/ordered_tile_${tile_index}.png" -o "tiles/tiles_${image}_pred/mask_tile_${tile_index}.png" --model $model
    done

    echo "Assembling Tiles for Image ${image}"
    mkdir -p predictions/$predict_output
    python utils/tile_assembly.py "tiles/tiles_${image}_pred" "predictions/$predict_output/${image}.png" --order $order
done

echo "All images processed successfully."
