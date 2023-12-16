import os
import shutil
import argparse
from random import shuffle

def split_data(base_path, split_ratio):
    print("Starting the data splitting process...")

    # Directories
    images_dir = os.path.join(base_path, 'training++/images')
    groundtruth_dir = os.path.join(base_path, 'training++/groundtruth')
    train_dir = os.path.join(base_path, 'train++')
    val_dir = os.path.join(base_path, 'val++')

    # Create train and val directories if they don't exist
    for directory in [train_dir, val_dir]:
        os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'groundtruth'), exist_ok=True)

    # List files and shuffle
    files = os.listdir(images_dir)
    shuffle(files)
    print(f"Total number of images: {len(files)}")

    # Calculate split index
    split_index = int(len(files) * split_ratio)
    print(f"Number of images in the training set: {split_index}")
    print(f"Number of images in the validation set: {len(files) - split_index}")

    # Function to copy files
    def copy_files(file_list, destination):
        for file in file_list:
            shutil.copy(os.path.join(images_dir, file), os.path.join(destination, 'images', file))
            shutil.copy(os.path.join(groundtruth_dir, file), os.path.join(destination, 'groundtruth', file))

    # Copying the files
    copy_files(files[:split_index], train_dir)  # For training
    copy_files(files[split_index:], val_dir)    # For validation
    print("Data copying completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data into training and validation sets.')
    parser.add_argument('--ratio', type=float, default=0.9, help='Train/Val split ratio (default: 0.9)')
    args = parser.parse_args()

    split_data('./data', args.ratio)
