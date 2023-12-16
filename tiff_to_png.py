from PIL import Image
import os

def convert_tiff_to_png(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            image_path = os.path.join(source_dir, filename)
            with Image.open(image_path) as img:
                target_path = os.path.join(target_dir, os.path.splitext(filename)[0] + ".png")
                img.save(target_path, "PNG")
                print(f"Converted '{filename}' to PNG and saved as '{target_path}'")

# Paths to your directories
train_dir = './data/tiff/val'
train_labels_dir = './data/tiff/val_labels'

# Convert files in each directory
convert_tiff_to_png(train_dir, train_dir.replace('tiff', 'png'))
convert_tiff_to_png(train_labels_dir, train_labels_dir.replace('tiff', 'png'))

# Indicating the process is complete
print("Conversion process completed.")
