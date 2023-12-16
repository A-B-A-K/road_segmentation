from PIL import Image
import os

def count_white_pixels(image):
    white_pixels = 0
    for pixel in image.getdata():
        # Check for white pixel, considering both RGB and RGBA formats
        if pixel[:3] == (255, 255, 255):
            white_pixels += 1
    return white_pixels

def remove_images_with_excess_white(train_dir, train_labels_dir, threshold=0.10):
    total_images_processed = 0
    images_removed = 0

    for filename in os.listdir(train_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(train_dir, filename)
            with Image.open(image_path) as img:
                total_pixels = img.size[0] * img.size[1]
                white_pixels = count_white_pixels(img)
                white_percentage = white_pixels / total_pixels

                print(f"Processing '{filename}' - White Pixel Percentage: {white_percentage*100:.2f}%")

                if white_percentage > threshold:
                    # Delete the image
                    os.remove(image_path)
                    print(f"Removed image: {filename}")

                    # Delete the corresponding mask
                    mask_path = os.path.join(train_labels_dir, filename)
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                        print(f"Removed corresponding mask: {filename}")

                    images_removed += 1
                total_images_processed += 1

    print(f"Total images processed: {total_images_processed}")
    print(f"Total images removed: {images_removed}")

# Paths to your directories (adjust as per your setup)
train_dir = './data/png/train'
train_labels_dir = './data/png/train_labels'

# Process the images
remove_images_with_excess_white(train_dir, train_labels_dir)
