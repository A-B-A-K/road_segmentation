from skimage import io, morphology
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Define the disc structuring element
def disc(n):
    circle_func = lambda i, j: ((i - n//2)**2 + (j - n//2)**2) <= (n//2)**2
    return np.fromfunction(circle_func, shape=(n,n)).astype(np.uint8)

# Define the square structuring element
def square(n):
    return np.ones((n, n)).astype(np.uint8)

# Define the smoothing function
def smoothing(img):
    b = disc(9)
    out_open = morphology.opening(img, footprint=b)
    b2 = square(30)
    out_close = morphology.closing(out_open, footprint=b2)
    binary_out = np.where(out_close >= 40, 255, 0).astype(np.uint8)
    return binary_out

# Directories
input_dir = './predictions/hue_best_bs_DB/'
output_dir = './predictions/hue_best_bs_DB_ref/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for image_path in glob.glob(input_dir + '*.png'):
    # Load the image
    image = io.imread(image_path)

    # Apply the smoothing function
    smoothed_image = smoothing(image)

    # Save the smoothed image in the output directory
    output_filepath = output_dir + os.path.basename(image_path)
    io.imsave(output_filepath, smoothed_image)
