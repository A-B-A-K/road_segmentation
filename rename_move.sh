#!/bin/bash

i_start=1
i_end=50

# Check if the original file exists
for i in $(seq $i_start $i_end)
do
    original_file_path="data/test_set_images/test_${i}/better_maskkk.png"
    new_file_name="pred_test_${i}.png"
    destination_directory="predictions/unet_e15_2x2"

    if [ ! -f "$original_file_path" ]; then
        echo "Error: Original file does not exist."
        echo "$original_file_path"
        exit 1
    fi

    # Check if the destination directory exists, if not, create it
    if [ ! -d "$destination_directory" ]; then
        echo "Destination directory does not exist. Creating it."
        mkdir -p "$destination_directory"
    fi

    # Construct the new file path
    new_file_path="$destination_directory/$new_file_name"

    # Rename (move) and copy the file
    cp "$original_file_path" "$new_file_path"

    # Report completion
    echo "File copied from $original_file_path to $new_file_path"
done