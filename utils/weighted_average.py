import os
import sys
from PIL import Image
import numpy as np

def calculate_weighted_average(images, weights):
    # Normalize the weights to sum to 1
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    
    # Calculate the weighted average image
    weighted_avg = np.zeros_like(images[0], dtype=float)
    for im, weight in zip(images, weights):
        weighted_avg += im * weight
    
    # Clip values to be in the range [0, 255] and convert to uint8
    return np.clip(weighted_avg, 0, 255).astype(np.uint8)

def main(directories, weights):
    # Base directory for predictions, relative to the current working directory of the script
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'predictions')

    # Directory for the weighted averages
    weighted_avg_dir = os.path.join(base_dir, 'weighted_avg_DBSCAN_sq')
    os.makedirs(weighted_avg_dir, exist_ok=True)

    # Assuming image names are consistent across folders, get the list from the first directory
    image_names = os.listdir(os.path.join(base_dir, directories[0]))

    # Process each image
    for image_name in image_names:
        images = []
        for directory in directories:
            image_path = os.path.join(base_dir, directory, image_name)
            if os.path.exists(image_path):
                images.append(np.array(Image.open(image_path).convert('L'), dtype=float))
            else:
                print(f"Image {image_name} not found in directory {directory}. Skipping this image.")
                continue
        
        if len(images) == len(directories):  # Ensure all directories had the image
            # Calculate the weighted average image
            weighted_avg_image = calculate_weighted_average(images, weights)
            
            # Save the weighted average image
            weighted_avg_img = Image.fromarray(weighted_avg_image)
            weighted_avg_img.save(os.path.join(weighted_avg_dir, image_name))

if __name__ == "__main__":
    args = sys.argv[1:]  # Exclude the script name itself
    
    if len(args) % 2 != 0:
        print("Usage: python weighted_avg.py dir1 weight1 dir2 weight2 ...")
        sys.exit(1)
    
    # Split arguments into directories and their corresponding weights
    directories = args[0::2]  # Take every second argument starting from 0
    weights = [float(w) for w in args[1::2]]  # Take every second argument starting from 1
    
    # Run the main function with the parsed directories and weights
    main(directories, weights)
