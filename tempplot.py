import matplotlib.pyplot as plt
import numpy as np

# Example data
#list1 = [1069, 1439, 1568, 1083]
list1 = [1290]
list2 = [503]

# Indices for the x-axis
indices = np.arange(len(list1))  # [0, 1, 2, 3, 4]
bar_width = 0.3

# Create the bar chart
plt.figure(figsize=(4, 5))
plt.bar(indices - bar_width/2, list1, width=bar_width, label='Root ROI plant #1 mean', color='skyblue')
plt.bar(indices + bar_width/2, list2, width=bar_width, label='Soil ROI plant #1 mean', color='salmon')

# Labeling
plt.xlabel('ROI')
plt.ylabel('AUC')
plt.title('Bar Chart Comparing MEAN Root and Soil AUC')
plt.xticks(indices)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()