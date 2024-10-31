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
kU = np.ones((15, 15), np.float32) / (15 * 15)

plt.figure(figsize=(6, 6))
plt.imshow(kU, cmap='gray', interpolation='nearest')
plt.title("15x15 Uniform Kernel")
plt.colorbar()
plt.axis('off')
plt.savefig(os.path.join(dir_path, "uniform_kernel_plot.png"), bbox_inches='tight', dpi=300)
plt.show()

# 3 - Apply kernel to image
imU = cv2.filter2D(img_gray, -1, kU)

# 4 - Gaussian blur from scratch
def kgauss(sigma, half_width):
    size = half_width * 2 + 1   # Size must be odd
    if size % 2 == 0:
        raise ValueError("Size must be odd.")

    ax = np.arange(-half_width, half_width + 1) # -half_width to half_width around center pixel
    xx, yy = np.meshgrid(ax, ax)    # 2D grid for kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= kernel.sum()  # Normaliza para volume unitário

    return kernel

# Gaussian Blur with sigma 5 and half width of 8
kG = kgauss(5, 8)

plt.figure(figsize=(6, 6))
plt.imshow(kG, cmap='gray', interpolation='nearest')
plt.title("Gaussian Kernel")
plt.colorbar()
plt.axis('off')
plt.savefig(os.path.join(dir_path, "gaussian_kernel_plot.png"), bbox_inches='tight', dpi=300)
plt.show()

# 5 - Apply Gaussian blur
imG = cv2.filter2D(img_gray, -1, kG)
cv2.imshow("blur_cv2", imG)
cv2.waitKey(0)

# Gaussian blur direct implementation
# imG = cv2.GaussianBlur(img_gray, (17, 17), 5)

# 6 - Show image
cv2.imshow("Image with uniform filter", imU)
cv2.waitKey(0)
cv2.imshow("Image with Gaussian filter", imG)
cv2.waitKey(0)

# Save images
cv2.imwrite(dir_path + "\\flowers4_uniform.png", imU)
cv2.imwrite(dir_path + "\\flowers4_gaussian.png", imG)