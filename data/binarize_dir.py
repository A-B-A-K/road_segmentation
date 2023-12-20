import os
from PIL import Image

directory_path = './data/training/groundtruth/'

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(directory_path, filename)
        img = Image.open(image_path).convert('L')
        binary_img = img.point(lambda x: 255 if x > 127 else 0, '1')
        # Overwrite the original image
        binary_img.save(image_path)
