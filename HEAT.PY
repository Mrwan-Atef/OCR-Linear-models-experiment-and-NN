import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = np.array([
    [0.78, 0.81, 0.80],  # 0
    [0.75, 0.81, 0.78],  # 1
    [0.75, 0.72, 0.74],  # 2
    [0.81, 0.77, 0.78],  # 3
    [0.74, 0.76, 0.75],  # 4
    [0.77, 0.77, 0.77],  # 5
    [0.79, 0.87, 0.83],  # 6
    [0.74, 0.71, 0.72],  # 7
    [0.72, 0.62, 0.67],  # 8
    [0.73, 0.76, 0.75],  # 9
])

fig, ax = plt.subplots(figsize=(1.1, 2))
im = ax.imshow(metrics,aspect='auto')

# Labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(['Precision', 'Recall', 'F1-score'])
ax.set_yticklabels(range(10))

# Annotate cells
for i in range(10):
    for j in range(3):
        ax.text(j, i, f"{metrics[i, j]:.2f}",
                ha="center", va="center", color="white")

plt.colorbar(im)
plt.title("Logistic Regression (OVA) — Performance Heatmap")
plt.tight_layout()
plt.show()
