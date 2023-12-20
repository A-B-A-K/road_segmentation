import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

original_img = mpimg.imread('./predictions/orig/pred_test_22.png')
hue_img = mpimg.imread('./predictions/hue_best_bs/pred_test_22.png')
saturation_img = mpimg.imread('./predictions/sat/pred_test_22.png')
contrast_img = mpimg.imread('./predictions/contr/pred_test_22.png')
brightness_img = mpimg.imread('./predictions/bright/pred_test_22.png')
weighted_avg_img = mpimg.imread('./predictions/wa_equal_test/pred_test_22.png') # Assuming the weighted average is reusing the original image path

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 8))

# Set the title of the figure
# fig.suptitle('Road Segmentation Predictions and Weighted Average', fontsize=16)

# Plot each image in its respective subplot
axs[0, 0].imshow(original_img, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

axs[0, 1].imshow(hue_img, cmap='gray')
axs[0, 1].set_title('Hue')
axs[0, 1].axis('off')

axs[0, 2].imshow(saturation_img, cmap='gray')
axs[0, 2].set_title('Saturation')
axs[0, 2].axis('off')

axs[1, 0].imshow(contrast_img, cmap='gray')
axs[1, 0].set_title('Contrast')
axs[1, 0].axis('off')

axs[1, 1].imshow(brightness_img, cmap='gray')
axs[1, 1].set_title('Brightness')
axs[1, 1].axis('off')

axs[1, 2].imshow(weighted_avg_img, cmap='gray')
axs[1, 2].set_title('Weighted Average')
axs[1, 2].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
visualization_path = 'road_segmentation_visualization_wa.png'
plt.savefig(visualization_path)

# Show the plot
plt.show()

# Return the path to the saved figure for download
visualization_path