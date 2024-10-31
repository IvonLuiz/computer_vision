import cv2
import os
import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
traffic_video_path = dir_path + "\\traffic_sequence.mpg"

video_capture = cv2.VideoCapture(traffic_video_path)

# Creating a video writer
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(dir_path + '\\traffic_diadic.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, 
                      (width, height),
                      isColor=False)

# Get first frame
ret, prev_frame = video_capture.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while video_capture.isOpened():
    # Reading each frame from the video stream
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Diference between two frames
    frame_diff = cv2.absdiff(prev_gray, frame_gray)
    out.write(frame_diff)
    cv2.imshow('Diference between frames', frame_diff)

    prev_gray = frame_gray
    # Sair ao pressionar a tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()