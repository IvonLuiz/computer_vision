import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random as rng

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
_, binary_img = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

imshow(binary_img)

# Image moments
M = cv2.moments(binary_img)

# a) Area
area = M['m00']
print(f"Area from object: {area}")

# b) Centroids
cX = M["m10"] / area
cY = M["m01"] / area
centroids = (int(cX), int(cY))

cv2.circle(img, centroids, 5, (255, 0, 0), -1)
imshow(img)

# c) Central moments and inertia matrix
mu20 = M["mu20"]
mu02 = M["mu02"]
mu11 = M["mu11"]

inertia_matrix = np.array([[mu20, mu11], [mu11, mu02]])
print(f"Inertia matrix: \n {inertia_matrix}")

# d) Equivalent ellipse
eigenvalues, eigenvectors = np.linalg.eig(inertia_matrix)

a = 4 * np.sqrt(eigenvalues[0] / area)  # major axis length
b = 4 * np.sqrt(eigenvalues[1] / area)  # minor axis length

axes = (int(a / 2), int(b / 2))

print(f"Major axis 'a': {a}\nMinor axis 'b': {b}")

# e) Orientation
theta_rad = 0.5 * np.arctan((2 * mu11) / (mu20 - mu02))
theta_dgr = np.degrees(theta_rad)
print(f"Orientation in degrees: {theta_dgr}")

# f) Plot
ellipse = cv2.ellipse(
    img,
    centroids,
    axes,
    theta_dgr,
    0,
    360,
    (255, 0, 0),
    2,
)

imshow(ellipse)