import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reading the image
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "yellowtargets.png")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----- Chromaticity -----
# Splitting into RGB channels
R, G, B = cv2.split(img)

# Computing the chromaticity planes
Y =  R + G + B
mask = Y > 0  # Valid where Y is not zero

r = np.zeros_like(R, dtype=float)
g = np.zeros_like(G, dtype=float)

r[mask] = R[mask] / Y[mask]
g[mask] = G[mask] / Y[mask]

# Rescale r and g to 0-255 for visualization and plot
x_rescaled = (r * 255).astype(np.uint8)
y_rescaled = (g * 255).astype(np.uint8)

plt.figure(figsize=(12, 6))

# Chromaticity r-plane
plt.subplot(1, 2, 1)
plt.imshow(x_rescaled, cmap="Reds")
plt.title("Chromaticity Plane - r (Red Proportion)")
plt.colorbar(label="Intensity (0-255)")

# Chromaticity g-plane
plt.subplot(1, 2, 2)
plt.imshow(y_rescaled, cmap="Greens")
plt.title("Chromaticity Plane - g (Green Proportion)")
plt.colorbar(label="Intensity (0-255)")

plt.tight_layout()
# plt.show()

# Flatten the chromaticity coordinates for clustering
chromaticity = np.stack([r.flatten(), g.flatten()], axis=1)

# Flatten the channels
# r_flat = R.flatten()
# g_flat = G.flatten()
# b_flat = B.flatten()

# # Normalize values to [0, 1] for colors
# r_norm = r_flat / 255.0
# g_norm = g_flat / 255.0
# b_norm = b_flat / 255.0

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(r_flat, g_flat, b_flat, c=np.stack([r_norm, g_norm, b_norm], axis=1), s=1)

# ax.set_xlabel("Red Channel")
# ax.set_ylabel("Green Channel")
# ax.set_zlabel("Blue Channel")
# ax.set_title("3D RGB Color Scatter Plot")

# plt.show()

