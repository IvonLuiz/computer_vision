import cv2
import matplotlib.pyplot as plt
import os


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
# imshow(img)

M = cv2.moments(gray_image)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

imshow(img) 
print(cX)
print(cY)

# print(M)