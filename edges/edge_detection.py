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

dir_path = os.path.dirname(os.path.realpath(__file__))

image = cv2.imread(dir_path + '/penguins.png')
cv2_imshow(image, "Original")

# Convert to gray scale
img_grau = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel operator
Kv = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
)
Ku = Kv.T
Iu = cv2.filter2D(img_grau, -1, Ku)
Iv = cv2.filter2D(img_grau, -1, Kv)

cv2_imshow(Iu, "Iu")
cv2_imshow(Iv, "Iv")
# img_sobel = cv2.filter2D(img_grau, -1, KV)