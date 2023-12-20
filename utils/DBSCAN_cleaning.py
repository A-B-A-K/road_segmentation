import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import argparse

def process_image(image_path, size_threshold, output_directory, prediction_subfolder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Find all non-zero pixels
    non_zero_coords = np.column_stack(np.where(image > 0))
    
    # Perform DBSCAN
    db = DBSCAN(eps=2, min_samples=4).fit(non_zero_coords)
    labels = db.labels_
    
    # Get unique labels
    unique_labels = set(labels)
    
    # Generate a color map for each cluster
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    
    # Create an RGB image with the same dimensions as the grayscale image
    image_colored = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Apply color to each cluster
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Ignore noise/blobs
            continue
        cluster_pixels = non_zero_coords[labels == label]
        image_colored[cluster_pixels[:, 0], cluster_pixels[:, 1]] = color[::-1]  # Reverse to BGR
    
    # Create a new mask for clusters with less than size_threshold pixels
    mask_small_clusters = np.zeros_like(image, dtype=bool)
    for label, coord in zip(labels, non_zero_coords):
        if label != -1 and np.sum(labels == label) < size_threshold:
            mask_small_clusters[coord[0], coord[1]] = True
    
    # Remove small clusters from the image
    image_no_small_clusters = image.copy()
    image_no_small_clusters[mask_small_clusters] = 0

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the image with small clusters removed
    cleaned_image_name = f"{filename_without_ext}.png"
    cleaned_image_path = os.path.join(prediction_subfolder, cleaned_image_name)
    cv2.imwrite(cleaned_image_path, image_no_small_clusters)

    plot_output_path = os.path.join(output_directory, f"{filename_without_ext}_plot.png")
    
    # Display the original, colored, and cleaned images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(image_colored)
    axs[1].set_title('Colored Clusters')
    axs[1].axis('off')
    
    axs[2].imshow(image_no_small_clusters, cmap='gray')
    axs[2].set_title('Without Small Clusters')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()

def main(input_path, output_path, apply_on, size_threshold=500, vizualize=False):

    # Base directory for utils, which is the parent directory of the script's location
    base_utils_dir = os.path.dirname(__file__)

    # Output directory for DBSCAN processed images
    
    output_directory = os.path.join(base_utils_dir, '..', 'DBSCAN_viz', str(size_threshold))
    os.makedirs(output_directory, exist_ok=True)

    # Subfolder within the predictions directory for DBSCAN processed images
    if apply_on == 'val':
        prediction_subfolder = os.path.join(base_utils_dir, '..', 'data/val/pred', output_path)
    if apply_on == 'test':
        prediction_subfolder = os.path.join(base_utils_dir, '..', 'predictions', output_path)

    os.makedirs(prediction_subfolder, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"The folder {input_path} does not exist.")
        return

    # Iterate over files in the given folder
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_path, filename)
            # print(f"Processing {filename} with threshold {size_threshold}...")
            
            process_image(image_path, size_threshold, output_directory, prediction_subfolder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a folder and remove small clusters.")
    parser.add_argument("input_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_path", type=str, help="Name of the folder that will host the cleaned images.")
    parser.add_argument('--type', choices=['test', 'val'], default='pred', help="Are you predicting on the test set or validation set.")
    parser.add_argument("--threshold", type=int, default=500, help="Size threshold for removing small clusters.")
    parser.add_argument('--viz', action='store_true', default=False, help="Visualize the cleaning (result will be stored in a folder called DBSCAN_viz)")
    
    args = parser.parse_args()
    
    main(args.input_path, args.output_path, args.type, args.threshold, args.viz)