from PIL import Image
import os
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# Define directories
orig_dirs = {
    'train': {
        'images': './data/train++/images/',
        'groundtruth': './data/train++/groundtruth/'
    },
    'val': {
        'images': './data/val++/images/',
        'groundtruth': './data/val++/groundtruth/'
    }
}

aug_dirs = {
    'train': {
        'images': './data/train_aug_v2++/images/',
        'groundtruth': './data/train_aug_v2++/groundtruth/'
    },
    'val': {
        'images': './data/val_aug_v2++/images/',
        'groundtruth': './data/val_aug_v2++/groundtruth/'
    }
}

# Create augmentation directories if they don't exist
for key in aug_dirs:
    for subkey in aug_dirs[key]:
        if not os.path.exists(aug_dirs[key][subkey]):
            os.makedirs(aug_dirs[key][subkey])

# List of transformations
transformations = ['original', 'flip', 'rotate', 'scale']

# Parameters for transformations
angles = [45, 90, 135, 180, 225, 270, 315]
directions = ['horizontal', 'vertical']
scales = [1.5, 2, 2.5, 3]

# Function to apply transformations
def apply_transformations(orig_image_dir, orig_gt_dir, aug_image_dir, aug_gt_dir, set_type):
    print(f"Starting augmentation for {set_type} set...")
    for image_filename in os.listdir(orig_image_dir):
        print(f"Augmenting {image_filename}...")

        label_filename = image_filename  # Assuming label has the same filename

        # Load the image and mask
        image = Image.open(os.path.join(orig_image_dir, image_filename))
        label = Image.open(os.path.join(orig_gt_dir, label_filename))

        for transformation in transformations:
            # Original
            if transformation == 'original':
                transf_image = image
                transf_label = label.convert("L").point(lambda x: 255 if x > 128 else 0)

                transf_image.save(os.path.join(aug_image_dir, f'o_{image_filename}'))
                transf_label.save(os.path.join(aug_gt_dir, f'o_{label_filename}'))

            # Rotate
            elif transformation == 'rotate':
                for angle in angles:
                    transf_image = image.rotate(angle)
                    transf_label = label.rotate(angle).convert("L").point(lambda x: 255 if x > 128 else 0)

                    transf_image.save(os.path.join(aug_image_dir, f'r{angle}_{image_filename}'))
                    transf_label.save(os.path.join(aug_gt_dir, f'r{angle}_{label_filename}'))

            # Flip
            elif transformation == 'flip':
                for direction in directions:
                    if direction == 'horizontal':
                        transf_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        transf_label = label.transpose(Image.FLIP_LEFT_RIGHT).convert("L").point(lambda x: 255 if x > 128 else 0)

                        transf_image.save(os.path.join(aug_image_dir, f'fh_{image_filename}'))
                        transf_label.save(os.path.join(aug_gt_dir, f'fh_{label_filename}'))

                    elif direction == 'vertical':
                        transf_image = image.transpose(Image.FLIP_TOP_BOTTOM)
                        transf_label = label.transpose(Image.FLIP_TOP_BOTTOM).convert("L").point(lambda x: 255 if x > 128 else 0)

                        transf_image.save(os.path.join(aug_image_dir, f'fv_{image_filename}'))
                        transf_label.save(os.path.join(aug_gt_dir, f'fv_{label_filename}'))

            # Scale
            elif transformation == 'scale':
                for scale in scales:
                    transf_image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                    transf_label = label.resize((int(label.size[0] * scale), int(label.size[1] * scale))).convert("L").point(lambda x: 255 if x > 128 else 0)

                    transf_image = transf_image.crop((0, 0, image.size[0], image.size[1]))
                    transf_label = transf_label.crop((0, 0, label.size[0], label.size[1]))

                    transf_image.save(os.path.join(aug_image_dir, f's{scale}_{image_filename}'))
                    transf_label.save(os.path.join(aug_gt_dir, f's{scale}_{label_filename}'))


        print(f"Completed augmentation for {image_filename}.")
    print(f"Augmentation completed for {set_type} set.")

# Apply transformations to both train and val datasets
for set_type in orig_dirs:
    apply_transformations(orig_dirs[set_type]['images'], orig_dirs[set_type]['groundtruth'], aug_dirs[set_type]['images'], aug_dirs[set_type]['groundtruth'], set_type)
