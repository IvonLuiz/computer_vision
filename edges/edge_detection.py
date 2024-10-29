import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def cv2_imshow(img, title=''):
    plt.figure(figsize=(8,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


# Reading image
dir_path = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(dir_path + '/penguins.png')
cv2_imshow(image, "Original")


# Check gray scale image
if len(image.shape) == 2:
    print("Image in gray scale.")
else:
    if np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2]):
        print("Imagem in gray scale (same values for R, G, B).")
    else:
        print("Image is colored.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Gaussian blur
image_blur = cv2.GaussianBlur(image, (5, 5), 1)
cv2_imshow(image_blur, "Blur")


# Sobel operator
Kv = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
)
Ku = Kv.T

# Convolution
Iu = cv2.filter2D(image_blur, -1, Ku)
Iv = cv2.filter2D(image_blur, -1, Kv)

cv2_imshow(Iu, "Iu")
cv2_imshow(Iv, "Iv")

# Combining
I = np.sqrt(np.square(Iu) + np.square(Iv))
I = (I/I.max()*255).astype('uint8')

cv2_imshow(I, "I")
