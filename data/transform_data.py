import os
import random
import shutil
from PIL import Image, ImageEnhance

def adjust_brightness(image, factor):
    """ Adjusts brightness of an image. """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """ Adjusts contrast of an image. """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_hue(image, factor):
    """ Adjusts hue of an image. """
    # Convert the image to HSV, adjust the hue, and convert back to RGB
    image = image.convert('HSV')
    h, s, v = image.split()
    h = h.point(lambda p: (p + factor * 255) % 255)
    image = Image.merge('HSV', (h, s, v)).convert('RGB')
    return image

def adjust_saturation(image, factor):
    """ Adjusts saturation of an image. """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def augment_images(input_dir, output_images_dir, transformation):
    """ Augments images in the input directory with the specified transformation and saves them to the output directory. """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    for img_name in os.listdir(input_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            image_path = os.path.join(input_dir, img_name)
            image = Image.open(image_path)

            # Apply the specified transformation
            if transformation == 'brightness':
                factor = random.uniform(-0.2, 0.2)
                image = adjust_brightness(image, 1 + factor)
            elif transformation == 'contrast':
                factor = random.uniform(0.0, 0.5)
                image = adjust_contrast(image, 1 + factor)
            elif transformation == 'hue':
                factor = random.uniform(-0.2, 0.2)
                image = adjust_hue(image, factor)
            elif transformation == 'saturation':
                factor = random.uniform(0.0, 0.5)
                image = adjust_saturation(image, 1 + factor)
            
            # Save the transformed image
            image.save(os.path.join(output_images_dir, img_name))

def create_augmented_dataset(base_path, transformations):
    """ Creates augmented datasets for each specified transformation. """
    images_path = os.path.join(base_path, 'images')
    groundtruth_path = os.path.join(base_path, 'groundtruth')

    for transformation in transformations:
        print(f"Transformation: {transformation}")
        output_base_path = os.path.join(base_path + '_transf', f'train_{transformation}')
        output_images_path = os.path.join(output_base_path, 'images')
        output_groundtruth_path = os.path.join(output_base_path, 'groundtruth')

        # Augment images
        augment_images(images_path, output_images_path, transformation)

        # Copy ground truth
        if not os.path.exists(output_groundtruth_path):
            shutil.copytree(groundtruth_path, output_groundtruth_path)

def main():
    base_path = './data/train_aug++'
    transformations = ['hue', 'saturation', 'contrast', 'brightness', 'original']

    create_augmented_dataset(base_path, transformations)

if __name__ == "__main__":
    main()