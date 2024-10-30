import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def cv2_imshow(img):
    plt.figure(figsize=(8,5))
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = dir_path + "\\lena.pgm"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2_imshow(image)

# Changing data type
img_norm = (image/255).astype('float32')
print(img_norm)
print(img_norm.min(), img_norm.max())
cv2_imshow(img_norm)

# Changing brightness
img_bright = np.clip(img_norm + 0.25, 0, 1)
cv2_imshow(img_bright)

# Changing contrast
img_contrast = np.clip(img_norm * 2, 0, 1)
cv2_imshow(img_contrast)

# Negative image
img_neg = 1 - img_norm
cv2_imshow(img_neg)

# Posterisation
N = 4
img_post = np.floor(img_norm * N) / N
cv2_imshow(img_post)