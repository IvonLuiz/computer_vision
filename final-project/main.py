from process_image import Tracker
import os
import cv2


video_name1 = "videoos/Video1_husky.mp4"
video_name2 = "videoos/video2_husky.mp4"

dir_path = os.path.dirname(os.path.realpath(__file__))

video_path1 = os.path.join(dir_path, video_name1)
video_path2 = os.path.join(dir_path, video_name2)


cap = cv2.VideoCapture(video_path1)
images = []

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:
        images.append(frame)

    # Break the loop
    else:
        print("All images read")
        break

reference_frame = images[1045]
kmin = 2
kmax = 15

processor = Tracker()
processor.train_kmeans(reference_frame, kmin, kmax)
processor.process_video(video_path1)
processor.process_video(video_path2)
