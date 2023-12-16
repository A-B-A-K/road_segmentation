from PIL import Image
import os

def partition_and_resize_image(image_path, output_dir, prefix, tile_size, final_size=(400, 400)):
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        for i in range(3):  # 3 rows
            for j in range(3):  # 3 columns
                box = (j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size)
                tile = img.crop(box).resize(final_size)
                tile.save(os.path.join(output_dir, f"{prefix}_{i}_{j}.png"))

def process_images(images_dir, masks_dir, output_images_dir, output_masks_dir):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)

    for filename in os.listdir(images_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)

            if os.path.exists(mask_path):
                print(f"Processing '{filename}'")
                prefix = os.path.splitext(filename)[0]
                tile_size = Image.open(image_path).size[0] // 3  # Assuming square images

                partition_and_resize_image(image_path, output_images_dir, prefix, tile_size)
                partition_and_resize_image(mask_path, output_masks_dir, prefix, tile_size)

# Paths to your directories
images_dir = './data/png/val'
masks_dir = './data/png/val_labels'
output_images_dir = './data/val_mit/images'
output_masks_dir = './data/val_mit/groundtruth'

# Process the images and masks
process_images(images_dir, masks_dir, output_images_dir, output_masks_dir)
