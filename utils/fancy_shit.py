import numpy as np
from PIL import Image
import os

def value_to_class(v, foreground_threshold=0.75):
    df = np.sum(v)
    if df > foreground_threshold * (v.size):
        return 1
    else:
        return 0

def process_and_save_image(image_path, save_directory):
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    
    # Define patch size and prepare array for the classified image
    patch_size = 16
    classified_image = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    
    # Process the image and classify each patch
    for i in range(0, image_array.shape[0], patch_size):
        for j in range(0, image_array.shape[1], patch_size):
            patch = image_array[i:i + patch_size, j:j + patch_size]
            patch_class = value_to_class(patch)
            classified_image[i:i + patch_size, j:j + patch_size] = patch_class * 255
    
    # Save the classified image to the specified directory
    save_path = os.path.join(save_directory, 'sub_mask_111.png')
    Image.fromarray(classified_image).save(save_path)
    print(f"Classified image saved to {save_path}")

# Example usage:
process_and_save_image('./data/test_set_images/test_1/mask_1.png', './data/test_set_images/test_1/')
