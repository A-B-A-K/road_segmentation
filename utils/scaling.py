# scaling.py
from PIL import Image
import argparse

def scale_image(image_path, output_path, scale_factor):
    with Image.open(image_path) as img:
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        scaled_img = img.resize(new_size)
        scaled_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Scale an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("output_path", type=str, help="Path where the scaled image will be saved")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale factor for the image")

    args = parser.parse_args()

    scale_image(args.image_path, args.output_path, args.scale)

if __name__ == "__main__":
    main()
