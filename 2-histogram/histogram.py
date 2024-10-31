import cv2
import matplotlib.pyplot as plt
import os 


dir_path = os.path.dirname(os.path.realpath(__file__))

def cv2_imshow(img):
    plt.figure(figsize=(8,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Get image
image_path = dir_path + "//castle.jpg"
image = cv2.imread(image_path)

cv2_imshow(image)

# Histogram
img_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.plot(img_hist)
plt.savefig(dir_path + "//castle_histogram.png")
plt.show()

M = max(img_hist)

# Get binary image
T = 180

_, binary_image = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

cv2_imshow(binary_image)
cv2.imwrite(dir_path + "//castle_binary.jpg", binary_image)