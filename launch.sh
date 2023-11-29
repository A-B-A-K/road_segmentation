#!/bin/bash

# Specify the start and end indices for your iteration
start_index=1
end_index=50  # Change this to the total number of images you have

for (( i=start_index; i<=end_index; i++ ))
do
    python utils/predict.py -i data/test_set_images/test_${i}/test_${i}.png -o predictions/pred_test_${i}.png --model checkpoints/checkpoint_epoch10.pth
done