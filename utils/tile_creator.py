from PIL import Image
import os
import argparse

def split_image_into_ordered_tiles(image_path, output_folder, order):
    """
    Takes an image and splits it into a grid of tiles based on the specified order, saving them separately.

    Args:
    image_path (str): The path to the image file.
    output_folder (str): The folder where the tiles will be saved.
    order (int): The order of the split (e.g., 2 for 2x2, 3 for 3x3).

    Returns:
    list: A list of paths to the saved image tiles.
    """
    upscale_factor = 2

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder + "_pred"):
        os.makedirs(output_folder + "_pred")

    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size
        tile_width, tile_height = width // order, height // order

        # Generate and save each tile
        tile_paths = []
        for i in range(order):
            for j in range(order):
                coords = (j * tile_width, i * tile_height, (j + 1) * tile_width, (i + 1) * tile_height)
                tile = img.crop(coords)
                new_size = (int(tile.width * upscale_factor), int(tile.height * upscale_factor))
                # Resize the image
                tile = tile.resize(new_size)

                tile_path = os.path.join(output_folder, f"ordered_tile_{i*order + j + 1}.png")
                tile.save(tile_path)
                tile_paths.append(tile_path)

        return tile_paths
    

def main():
    parser = argparse.ArgumentParser(description="Split an image into a grid of ordered tiles.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("output_folder", type=str, help="Folder where the tiles will be saved")
    parser.add_argument("-o", "--order", type=int, default=2, help="The order of the split (e.g., 2 for 2x2, 3 for 3x3)")

    args = parser.parse_args()

    split_image_into_ordered_tiles(args.image_path, args.output_folder, args.order)


if __name__ == "__main__":
    main()