import cv2
import os
import matplotlib.pyplot as plt

# 1 - Reading image
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "flowers9.png")
img = cv2.imread(img_path)

# Displaying image in gray and double
# img = cv2.cvtColor(img_original)
# cv2.imshow("Image colored", img)
cv2.waitKey(0)

(B, G, R) = cv2.split(img)

# cv2.imshow('blue channel', B) 
# cv2.imshow('green channel', G) 
# cv2.imshow('red channel', R) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# Flatten the 2-D arrays of the RGB channels into 1-D
R_pixels = R.flatten()
G_pixels = G.flatten()
B_pixels = B.flatten()


plt.hist(R_pixels, bins=256)
plt.title("Red channel histogram")
plt.show()

plt.hist(G_pixels, bins=256)
plt.title("Green channel histogram")
plt.show()

plt.hist(B_pixels, bins=256)
plt.title("Blue channel histogram")
plt.show()
