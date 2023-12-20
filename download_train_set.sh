#!/bin/bash

# Google Drive file ID
FILE_ID="1Nef1sKcbtGfjhB7T9ED7zSdE6XE2KqMO"

# Target directory
TARGET_DIR="./data"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Temporary cookie file
COOKIE_FILE=$(mktemp)

# First request to get the confirmation token
CONFIRMATION=$(wget --quiet --save-cookies $COOKIE_FILE --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

# Download the file with the confirmation token
wget --load-cookies $COOKIE_FILE "https://drive.google.com/uc?export=download&confirm=$CONFIRMATION&id=$FILE_ID" -O downloaded.zip

# Clean up the cookie file
rm $COOKIE_FILE

# Unzip the file
unzip downloaded.zip -d "$TARGET_DIR"

mv "$TARGET_DIR/cil-road_segmentation-master-data-external-epfl-processed/data/external/epfl/processed" "$TARGET_DIR/training"
rm -r "$TARGET_DIR/cil-road_segmentation-master-data-external-epfl-processed"


# Checking the directory structure
if [ -d "$TARGET_DIR/training/groundtruth" ] && [ -d "$TARGET_DIR/training/images" ]; then
    echo "Folders are correctly placed"
else
    echo "Folders are not in the expected structure in the zip file"
fi

# Remove the downloaded zip file
rm downloaded.zip

echo "Download and extraction complete."
