import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Confusion matrix data
confusion_matrix = np.array([[50, 3, 2],
                             [4, 48, 3],
                             [1, 2, 52]])

# Normalize the confusion matrix
row_sums = confusion_matrix.sum(axis=1, keepdims=True)
proportional_matrix = confusion_matrix / row_sums

# Class labels
labels = ['Background', 'Method', 'Comparative']

# Plotting with enhanced colors
plt.figure(figsize=(8, 6))
sns.heatmap(proportional_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
            xticklabels=labels, yticklabels=labels, cbar=True, annot_kws={"size": 12})

plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)

# Rotate the tick labels for better visibility
plt.xticks(rotation=0, fontsize=16)
plt.yticks(rotation=90, fontsize=16)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('matrix.svg')
plt.savefig('matrix.png')
# Show the plot
plt.show()
