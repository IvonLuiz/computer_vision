import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 1 - Reading image
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "flowers4.png")
img_original = cv2.imread(img_path)

# Displaying image in gray and double
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image gray double", img_gray)
cv2.waitKey(0)

# 2 - Make 15x15 kernel with volume one
Ku = np.ones((15, 15), np.float32) / (15 * 15)

plt.figure(figsize=(6, 6))
plt.imshow(Ku, cmap='gray', interpolation='nearest')
plt.title("15x15 Uniform Kernel")
plt.colorbar()
plt.axis('off')
plt.show()

# 3 - Apply kernel to image
im_u = cv2.filter2D(img_gray, -1, Ku)

