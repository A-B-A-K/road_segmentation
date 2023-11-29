#!/bin/bash

# Start and end indices for image and tile iteration
start_image_i=1
end_image_i=50
start_tile_i=1
end_tile_i=4

for image_index in $(seq $start_image_i $end_image_i)
do
    echo "Processing Image $image_index"
    python data/test_set_images/tile_creator.py "data/test_set_images/test_${image_index}/test_${image_index}.png" "data/test_set_images/test_${image_index}/"

    for tile_index in $(seq $start_tile_i $end_tile_i)
    do
        echo "Predicting Tile $tile_index for Image $image_index"
        python utils/predict.py -i "data/test_set_images/test_${image_index}/ordered_tile_${tile_index}.png" -o "data/test_set_images/test_${image_index}/mask_tile${tile_index}.png" --model checkpoints/checkpoint_epoch15.pth
    done

    echo "Assembling Tiles for Image $image_index"
    python data/test_set_images/tile_assembly.py "data/test_set_images/test_${image_index}/mask_tile1.png" "data/test_set_images/test_${image_index}/mask_tile2.png" "data/test_set_images/test_${image_index}/mask_tile3.png" "data/test_set_images/test_${image_index}/mask_tile4.png" "data/test_set_images/test_${image_index}/better_maskkk.png"
done

echo "All images processed successfully."
