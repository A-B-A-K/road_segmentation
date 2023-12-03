#!/bin/bash

# Start and end indices for image and tile iteration
order=1
start_image_i=6
end_image_i=6
start_tile_i=1
end_tile_i=$((order * order))
model=checkpoints/checkpoint_epoch15.pth
predict_output="new_predictions_one"

for image_index in $(seq $start_image_i $end_image_i)
do
    echo "Processing Image $image_index"
    python utils/tile_creator.py "data/test_set_images/test_${image_index}/test_${image_index}.png" "tiles/tiles_${image_index}" --order $order

    for tile_index in $(seq $start_tile_i $end_tile_i)
    do
        echo "Predicting Tile $tile_index for Image $image_index"
        python predict.py -i "tiles/tiles_${image_index}/ordered_tile_${tile_index}.png" -o "tiles/tiles_${image_index}_pred/mask_tile_${tile_index}.png" --model $model
    done

    echo "Assembling Tiles for Image $image_index"
    mkdir -p predictions/$predict_output
    python utils/tile_assembly.py "tiles/tiles_${image_index}_pred" "predictions/$predict_output/pred_test_${image_index}.png" --order $order
done

echo "All images processed successfully."
