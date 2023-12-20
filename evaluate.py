import os
import glob
import cv2
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

val = 'val++'
models = ['orig', 'hue', 'sat', 'contr', 'bright']

model = 'bin_wa_uneq4_ohcbs_db_0.9'

# Define paths
image_folder = f'./data/{val}/images'
ground_truth_folder = f'./data/{val}/groundtruth'
predictions_folder = f'./data/{val}/pred/{model}'

# Check if paths exist
if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder) or not os.path.exists(predictions_folder):
    raise FileNotFoundError("One or more directories do not exist.")

# Function to load images
def load_images(folder):
    images = {}
    for filepath in glob.glob(os.path.join(folder, '*.png')):
        filename = os.path.basename(filepath)
        images[filename] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return images

# Load images, ground truths, and predictions
images = load_images(image_folder)
ground_truths = load_images(ground_truth_folder)
predictions = load_images(predictions_folder)

# Ensure all sets have the same files
if set(images.keys()) != set(ground_truths.keys()) or set(images.keys()) != set(predictions.keys()):
    raise ValueError("Mismatch in filenames across folders.")

# Flatten and calculate F1 score and accuracy
ground_truth_vals = np.array([ground_truths[key].flatten() for key in sorted(ground_truths.keys())]).flatten()
predictions_vals = np.array([predictions[key].flatten() for key in sorted(predictions.keys())]).flatten()

f1 = f1_score(ground_truth_vals, predictions_vals, pos_label=255)
accuracy = accuracy_score(ground_truth_vals, predictions_vals)

print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
