import cv2
import os
import matplotlib.pyplot as plt

def cv2_imshow(img):
    plt.figure(figsize=(8,5))
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = dir_path + "\\lena.pgm"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# cv2_imshow(image)

# Changing data type
img_norm = (image/255).astype('float32')
print(img_norm)
print(img_norm.min(), img_norm.max())
cv2_imshow(img_norm)

# Changing brightness
img_bright = img_norm + 0.25
print(img_bright)
print(img_bright.min(), img_bright.max())
cv2_imshow(img_bright)
