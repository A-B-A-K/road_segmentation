import os, sys, argparse
from PIL import Image
import numpy as np

def apply_threshold(image_array, threshold=128):
    return (image_array > threshold).astype(np.uint8) * 255

def main(input_path, output_path, apply_on, threshold_value):
    # Base directory for predictions, relative to the current working directory of the script
    if apply_on == 'test':
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'predictions')
    if apply_on == 'val':
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data/val/pred')

    # Directory for the weighted averages
    weighted_avg_dir = os.path.join(base_dir, input_path)

    # Directory for the binary masks
    binary_masks_dir = os.path.join(base_dir, output_path)
    os.makedirs(binary_masks_dir, exist_ok=True)

    # Process each weighted average image
    for image_name in os.listdir(weighted_avg_dir):
        image_path = os.path.join(weighted_avg_dir, image_name)
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        
        binary_mask = apply_threshold(image_array, threshold_value)
        
        binary_mask_img = Image.fromarray(binary_mask)
        binary_mask_img.save(os.path.join(binary_masks_dir, image_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold weighted average images in order to binarize them.")
    parser.add_argument("input_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_path", type=str, help="Name of the folder that will host the binary images.")
    parser.add_argument('--type', choices=['test', 'val'], default='pred', help="Are you predicting on the test set or validation set.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification as a background or road pixel.")
    
    args = parser.parse_args()
    
    main(args.input_path, args.output_path, args.type, args.threshold*256)
