import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

# Function to process the images
def process_images(input_dir, output_dir, threshold):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all the image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        binary_image = cv2.imread(image_path, 0)

        # Check if the image is binary
        if np.unique(binary_image).tolist() != [0, 255]:
            _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        # Perform Euclidean distance transform
        distance_transform = distance_transform_edt(binary_image)

        # Normalize the distance transform image to get values between 0 and 1
        normalized_distance_transform = distance_transform / distance_transform.max()

        # Create binary mask with threshold
        binary_mask = (normalized_distance_transform >= threshold).astype(np.uint8) * 255

        # Construct the output path
        output_image_path = os.path.join(output_dir, image_file)

        # Save the binary mask image
        plt.imsave(output_image_path, binary_mask, cmap='gray')

    return [os.path.join(output_dir, f) for f in image_files]

def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description='Process binary ground truth images.')
    parser.add_argument('input_dir', type=str, help='Directory with binary ground truth images')
    parser.add_argument('output_dir', type=str, help='Directory where transformed images will be stored')
    parser.add_argument('--threshold', type=float, help='Threshold for binary mask creation', default=0.2)

    # Parse arguments
    args = parser.parse_args()

    # Run the processing function with the arguments
    print("Start...")
    output_images = process_images(args.input_dir, args.output_dir, args.threshold)
    print(f"Processed images stored in {args.output_dir}:")

    for image_path in output_images:
        print(image_path)

if __name__ == "__main__":
    main()
