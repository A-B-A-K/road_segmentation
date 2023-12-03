from PIL import Image
import sys, os
import argparse

def combine_tiles_into_image(tile_paths, output_path, order):
    """
    Combines image tiles back into a single image based on the specified order.

    Args:
    tile_paths (list): A list of paths to the image tiles.
    output_path (str): The path where the combined image will be saved.
    order (int): The order of the split (e.g., 2 for 2x2, 3 for 3x3).

    Returns:
    str: The path to the saved combined image.
    """
    downscale_factor = 0.5
    
    tiles = [Image.open(tile) for tile in tile_paths]

    # Assuming all tiles are of the same size
    tile_width, tile_height = tiles[0].size

    # Create a new image with width and height based on the order
    new_image = Image.new('RGB', (tile_width * order, tile_height * order))

    # Place each tile in the new image in the correct position
    for i, tile in enumerate(tiles):
        x = (i % order) * tile_width
        y = (i // order) * tile_height
        new_image.paste(tile, (x, y))

    new_size = (int(new_image.width * downscale_factor), int(new_image.height * downscale_factor))
    # Resize the image
    new_image = new_image.resize(new_size)

    # Save the combined image
    new_image.save(output_path)

    return output_path

def generate_tile_paths(input_folder, order):
    """
    Generates a list of tile image paths based on the input folder and order.

    Args:
    input_folder (str): The folder containing the tile images.
    order (int): The order of the split (e.g., 2 for 2x2, 3 for 3x3).

    Returns:
    list: A list of paths to the image tiles.
    """
    tile_paths = []
    for i in range(order**2):
        tile_path = os.path.join(input_folder, f"mask_tile_{i + 1}.png")
        tile_paths.append(tile_path)
    return tile_paths

def main():
    parser = argparse.ArgumentParser(description="Combine image tiles into a single image.")
    parser.add_argument("input_folder", type=str, help="Folder containing the image tiles")
    parser.add_argument("output_path", type=str, help="Path where the combined image will be saved")
    parser.add_argument("-o", "--order", type=int, default=2, help="The order of the split (e.g., 2 for 2x2, 3 for 3x3)")

    args = parser.parse_args()

    tile_paths = generate_tile_paths(args.input_folder, args.order)
    combine_tiles_into_image(tile_paths, args.output_path, args.order)

if __name__ == "__main__":
    main()
