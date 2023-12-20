#!/bin/bash

# Google Drive file ID
FILE_ID="194Mr-s12Nm3Wh7crM96-v-8Lx3CJSuXz"

# Target directory
TARGET_DIR="./checkpoints"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"
mkdir -p "./weights"

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
mv "$TARGET_DIR/best/"* "./weights/"
rm -r "$TARGET_DIR"

# Remove the downloaded zip file
rm downloaded.zip

echo "Download and extraction complete."
