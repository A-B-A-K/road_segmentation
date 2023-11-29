#!/bin/bash

# Specify the start and end indices for your tiles
start_index_t=1
end_index_t=4  # Change this to the total number of tiles you have



for (( i=start_index_t; i<=end_index_t; i++ ))
do
    python utils/predict.py -i data/test_set_images/test_2/ordered_tile_${i}.png -o data/test_set_images/test_2/mask_tile${i}.png --model checkpoints/checkpoint_epoch15.pth
done



