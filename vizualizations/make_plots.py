import matplotlib.pyplot as plt
import numpy as np

# Data
weight_combinations = ['Equal', 'F1-Dominant', 'Exp. Extreme', 'Inverse Error', 'Exp. Extreme++', 'Performance Based']
dbscan = False
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
if dbscan:
    equal_data = [0.8388, 0.8913, 0.903, 0.8914, 0.8558]
    uneq1_data = [0.8852, 0.9005, 0.9098, 0.9095, 0.8558]
    uneq2_data = [0.8388, 0.9063, 0.9112, 0.9063, 0.8558]
    uneq3_data = [0.8514, 0.9018, 0.912, 0.9009, 0.8558]
    uneq4_data = [0.8902, 0.9096, 0.9096, 0.9112, 0.8809]
    uneq5_data = [0.8388, 0.9045, 0.9098, 0.8911, 0.8558]
else:
    equal_data = [0.835, 0.8905, 0.904, 0.8929, 0.8568]
    uneq1_data = [0.8842, 0.9001, 0.9103, 0.9068, 0.8568]
    uneq2_data = [0.835, 0.9056, 0.911, 0.9077, 0.8568]
    uneq3_data = [0.8493, 0.9009, 0.9119, 0.9026, 0.8568]
    uneq4_data = [0.8892, 0.9092, 0.9092, 0.911, 0.8817]
    uneq5_data = [0.835, 0.909, 0.9103, 0.8928, 0.8568]

x = np.arange(len(weight_combinations))  # the label locations
bar_width = 0.15  # the width of the bars
palette = ['#ed899d', '#ffb98b', '#feedaa', '#a5dd8b', '#9ec6ee']

# Setting up the figure and axes
fig, ax = plt.subplots(figsize=(12, 4))  # Longer figure

# Adjusted bar width and spacing
bar_width = 0.12
spacing = 0.05  # Additional space between groups

# Creating bars for each threshold with increased spacing between groups
for i, threshold in enumerate(thresholds):
    ax.bar(x + i * bar_width + (x * spacing), 
           [equal_data[i], uneq1_data[i], uneq2_data[i], uneq3_data[i], uneq4_data[i], uneq5_data[i]], 
           bar_width, label=f'{threshold}', color=palette[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Weight Combinations')
ax.set_ylabel('F1-score')
ax.set_ylim([0, 1])
ax.set_yticks(np.arange(0, 1.1, 0.1))
# ax.set_title('F1-scores by Weight Combination and Threshold')
ax.set_xticks(x + bar_width * 2 + (x * spacing))
ax.set_xticklabels(weight_combinations)
ax.legend(title='Thresholds', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adding a grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')  # Grid for y-axis

# Adjust layout for clarity
fig.tight_layout()

# Saving the figure
output_filepath = 'wa_dbscan0.png'
fig.savefig(output_filepath)
