import numpy as np
import cv2
import os
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "church.jpg")

# Read image
src = cv2.imread(img_path,cv2.IMREAD_COLOR)
# cv2.imshow('lanes',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
edges = cv2.Canny(gray, 50, 200, None, 3) # find the edges in the image using canny detector

# Copy edges to the images that will display the results in BGR
edgesT = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edgesP = np.copy(edgesT)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 180, None, 0, 0)
print(lines)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        
        a = math.cos(theta)
        b = math.sin(theta)
        
        x0 = a * rho
        y0 = b * rho
        
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(edgesT, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# Detect points that form a line
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 68, minLineLength=15, maxLineGap=250)
# lines = cv2.HoughLines(edges, 1, np.pi/180, 150, None,0 ,0)

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, None, 50, 10)
print(linesP)

if linesP is not None:
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(edgesP, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)

# Show result
cv2.imshow("Source", src)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", edgesT)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", edgesP)

cv2.waitKey(0)
cv2.destroyAllWindows()