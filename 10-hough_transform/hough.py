import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "church.jpg")

# Read image
img = cv2.imread(img_path,cv2.IMREAD_COLOR)
# cv2.imshow('lanes',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
edges = cv2.Canny(gray, 50, 200) # find the edges in the image using canny detector

# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 68, minLineLength=15, maxLineGap=250)
print(lines)

# Draw lines on the image
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Show result
print("Line Detection using Hough Transform")
cv2.imshow('lanes',img)
cv2.waitKey(0)
cv2.destroyAllWindows()