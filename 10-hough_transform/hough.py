import numpy as np
import cv2
import os
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "church.jpg")

# Read image
src = cv2.imread(img_path,cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
edges = cv2.Canny(gray, 50, 200, None, 3) # find the edges in the image using canny detector

# Copy edges to the images that will display the results in BGR
imgHoughLines = np.copy(src)
imgHoughLinesProb = np.copy(src)

# Standard Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 180, None, 0, 0)
print(lines)

# Lines has rho and theta values
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]    # distance from origin to the line
        theta = lines[i][0][1]  # angle from origin to the line
        
        a = math.cos(theta)
        b = math.sin(theta)
    
        # coordinates of the line closer to the origin
        x0 = a * rho
        y0 = b * rho            # closer to the origin
        
        # extreme points from the line
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(imgHoughLines, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# Probabilistic Hough Transform
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, None, 50, 10)
print(linesP)

if linesP is not None:
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(imgHoughLinesProb, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)

# Show results
cv2.imshow("Source", src)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", imgHoughLines)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", imgHoughLinesProb)

cv2.waitKey(0)
cv2.destroyAllWindows()