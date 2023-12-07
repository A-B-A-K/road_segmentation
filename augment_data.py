# Define directories
from PIL import Image, ImageEnhance, ImageOps
import random
import os


orig_image_dir = 'data/training/images/'
orig_gt_dir = 'data/training/groundtruth/'
aug_image_dir = 'data/training_augmented/images/'
aug_gt_dir = 'data/training_augmented/groundtruth/'

# Create the directories if they don't exist
if not os.path.exists(aug_image_dir):
    os.makedirs(aug_image_dir)

if not os.path.exists(aug_gt_dir):
    os.makedirs(aug_gt_dir)

# List of transformations to apply
#transformations = ['original', 'flip', 'rotate', 'scale']
transformations = ['original', 'rotate', 'flip', 'scale', 'brightness', 'contrast', 'hue', 'saturation']


# Parameters for transformations
#angles = [45, 90, 135, 180, 225, 270, 315]
angles = [90, 180, 270]
directions = ['horizontal', 'vertical']
scales = [1.25, 1.5, 1.75, 2]

# Loop over all images
for image_filename in os.listdir(orig_image_dir):
    label_filename = image_filename  # Mask has the same filename as the image

    # Load the image and mask
    image = Image.open(os.path.join(orig_image_dir, image_filename))
    label = Image.open(os.path.join(orig_gt_dir, label_filename))

    # Apply the transformations
    for transformation in transformations:
        if transformation == 'original':
            transf_image = image
            transf_label = label

            # Save the transformed images and masks
            transf_image.save(os.path.join(aug_image_dir, f'o_{image_filename}'))
            transf_label.save(os.path.join(aug_gt_dir, f'o_{label_filename}'))
            
        if transformation == 'rotate':
            for angle in angles:
                transf_image = image.rotate(angle)
                transf_label = label.rotate(angle)

                # Save the transformed images and masks
                transf_image.save(os.path.join(aug_image_dir, f'r{angle}_{image_filename}'))
                transf_label.save(os.path.join(aug_gt_dir, f'r{angle}_{label_filename}'))

        elif transformation == 'flip':
            for direction in directions:
                if direction == 'horizontal':
                    transf_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    transf_label = label.transpose(Image.FLIP_LEFT_RIGHT)

                    # Save the transformed images and masks
                    transf_image.save(os.path.join(aug_image_dir, f'fh_{image_filename}'))
                    transf_label.save(os.path.join(aug_gt_dir, f'fh_{label_filename}'))

                elif direction == 'vertical':
                    transf_image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    transf_label = label.transpose(Image.FLIP_TOP_BOTTOM)

                    # Save the transformed images and masks
                    transf_image.save(os.path.join(aug_image_dir, f'fv_{image_filename}'))
                    transf_label.save(os.path.join(aug_gt_dir, f'fv_{label_filename}'))
                
        elif transformation == 'scale':
            for scale in scales:
                transf_image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                transf_label = label.resize((int(label.size[0] * scale), int(label.size[1] * scale)))

                # Crop the images to the original size
                transf_image = transf_image.crop((0, 0, image.size[0], image.size[1]))
                transf_label = transf_label.crop((0, 0, label.size[0], label.size[1]))

                # Save the transformed images and masks
                transf_image.save(os.path.join(aug_image_dir, f's{scale}_{image_filename}'))
                transf_label.save(os.path.join(aug_gt_dir, f's{scale}_{label_filename}'))
        
        elif transformation == 'brightness':
            factor = 1 + random.uniform(-0.2, 0.2)  # Adjust the factor in the range of 0.8 to 1.2
            enhancer = ImageEnhance.Brightness(image)
            transf_image = enhancer.enhance(factor)
            # No change to the label as it's not affected by brightness
            transf_label = label
            transf_image.save(os.path.join(aug_image_dir, f'b_{factor}_{image_filename}'))
            transf_label.save(os.path.join(aug_gt_dir, f'b_{factor}_{label_filename}'))

        elif transformation == 'contrast':
            factor = random.uniform(0.0, 0.5)  # Adjust the factor in the range of 0.0 to 0.5
            enhancer = ImageEnhance.Contrast(image)
            transf_image = enhancer.enhance(factor)
            # No change to the label as it's not affected by contrast
            transf_label = label
            transf_image.save(os.path.join(aug_image_dir, f'c_{factor}_{image_filename}'))
            transf_label.save(os.path.join(aug_gt_dir, f'c_{factor}_{label_filename}'))

        elif transformation == 'hue':
            factor = random.uniform(-0.2, 0.2)  # Adjust the factor in the range of -0.2 to 0.2
            hsv_image = image.convert('HSV')
            h, s, v = hsv_image.split()
            h = h.point(lambda i: (i + int(factor * 255)) % 256)
            transf_image = Image.merge('HSV', (h, s, v)).convert('RGB')
            # No change to the label as it's not affected by hue
            transf_label = label
            transf_image.save(os.path.join(aug_image_dir, f'h_{factor}_{image_filename}'))
            transf_label.save(os.path.join(aug_gt_dir, f'h_{factor}_{label_filename}'))

        elif transformation == 'saturation':
            factor = random.uniform(0.0, 0.5)  # Adjust the factor in the range of 0.0 to 0.5
            enhancer = ImageEnhance.Color(image)
            transf_image = enhancer.enhance(factor)
            # No change to the label as it's not affected by saturation
            transf_label = label
            transf_image.save(os.path.join(aug_image_dir, f'sat_{factor}_{image_filename}'))
            transf_label.save(os.path.join(aug_gt_dir, f'sat_{factor}_{label_filename}'))
