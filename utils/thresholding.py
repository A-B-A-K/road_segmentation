import os, sys
from PIL import Image
import numpy as np

def apply_threshold(image_array, threshold=128):
    # Apply a threshold to the image array
    return (image_array > threshold).astype(np.uint8) * 255

def main(threshold_value):
    # Base directory for predictions, relative to the current working directory of the script
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'predictions')

    # Directory for the weighted averages
    weighted_avg_dir = os.path.join(base_dir, 'weighted_avg_DBSCAN')

    # Directory for the binary masks
    binary_masks_dir = os.path.join(base_dir, f'binary_masks_DBSCAN_c_{threshold_value/256}')
    os.makedirs(binary_masks_dir, exist_ok=True)

    # Process each weighted average image
    for image_name in os.listdir(weighted_avg_dir):
        # Load the weighted average image
        image_path = os.path.join(weighted_avg_dir, image_name)
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        
        # Apply the threshold to create a binary mask
        binary_mask = apply_threshold(image_array, threshold_value)
        
        # Save the binary mask image
        binary_mask_img = Image.fromarray(binary_mask)
        binary_mask_img.save(os.path.join(binary_masks_dir, image_name))

if __name__ == "__main__":
    # Accept threshold value as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <threshold_value>")
        sys.exit(1)
    
    try:
        threshold_value = float(sys.argv[1])
    except ValueError:
        print("Threshold value must be a number.")
        sys.exit(1)
    
    main(threshold_value*256)
