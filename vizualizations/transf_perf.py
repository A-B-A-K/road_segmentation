import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Original', 'Hue', 'Saturation', 'Contrast', 'Brightness']
f1_scores = [0.9009, 0.9092, 0.8368, 0.8598, 0.8682]
accuracy = [0.9636, 0.9669, 0.9414, 0.9480, 0.9512]

# Number of categories
n_categories = len(categories)

# The spacing between pairs is increased by adjusting the group spacing
group_space = 1.5  # Increased space between each group

# X locations for the groups
ind = np.arange(0, n_categories * group_space, group_space)
width = 0.3  # width of the bars

fig, ax = plt.subplots(figsize=(12,4))

# Placing the bars
f1_bars = ax.bar(ind - width/2, f1_scores, width, label='F1 Score', color='skyblue')
acc_bars = ax.bar(ind + width/2, accuracy, width, label='Accuracy', color='lightcoral')

# Adding smaller, grey numbers with precision of :.3 on top of each bar
for bar in f1_bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8,  # Smaller font size
                color='grey')  # Grey color

for bar in acc_bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8,  # Smaller font size
                color='grey')  # Grey color

# Labels, title, and custom x-axis tick labels
ax.set_ylabel('Performance')
ax.set_xticks(ind)
ax.set_xticklabels(categories)

# Hide the right and top spines and add grid
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.grid(True, color='#888586', linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Positioning the legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
# Adjust layout for clarity
fig.tight_layout()

# Save the figure
output_filepath = 'transf_viz.png'
fig.savefig(output_filepath)

output_filepath

