import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def imshow(img):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Reading image
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "aviao_ed.png")
img = cv2.imread(img_path)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

# imshow(binary_img)
# Image moments
M = cv2.moments(gray_image)

# a) Area
area = M['m00']
print(f"Area from object: {area}")

# b) Centroids
cX = int(M["m10"] / area)
cY = int(M["m01"] / area)
centroids = (int(cX), int(cY))
cv2.circle(img, centroids, 5, (255, 0, 0), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

imshow(img)

# c) Central moments and inertia matrix
mu20 = M["mu20"]
mu02 = M["mu02"]
mu11 = M["mu11"]

inertia_matrix = np.array([[mu20, mu11], [mu11, mu02]])
print(inertia_matrix)

