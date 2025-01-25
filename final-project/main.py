import numpy as np
import cv2
import os

video_name = "Video1_husky.mp4"

dir_path = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(dir_path, video_name)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
            
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()