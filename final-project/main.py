import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass
from scipy.ndimage import label

from utils import *

video_name = "Video1_husky.mp4"

dir_path = os.path.dirname(os.path.realpath(__file__))

video_path = os.path.join(dir_path, video_name)
cap = cv2.VideoCapture(video_path)
images = []

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:
        images.append(frame)
    #     # Display the resulting frame
    #     cv2.imshow('Frame', frame)
            
    #     # Press Q on keyboard to exit
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break

    # Break the loop
    else:
        print("All images read")
        break

# One frame for reference
ref_img = images[1045] # 670 moment that pass on a obstacle

# rgb_splitter(ref_img)

img_blur = ref_img
img_blur = cv2.GaussianBlur(ref_img, (7, 7), 2.0)

Y = img_blur.sum(axis=2) # R + G + B
r = img_blur[:,:,0] / Y
g = img_blur[:,:,1] / Y

one_matrix = np.ones_like(float, shape=r.shape)
b = one_matrix- (r + g)

img_chromatic_rgb = cv2.merge([np.uint8(r*255),
                               np.uint8(g*255),
                               np.uint8(b*255)]) 

# cv2.imshow("Chromatic RGB img", img_chromatic_rgb)
# RG_Chroma_plotter(r, g)

# Flatten the chromaticity coordinates for clustering
chromaticity = np.float32(np.stack([r.flatten(), g.flatten()], axis=1))

## Search for optimal value of K clusters with elbow method
kmin = 2
kmax = 15

print("Calculating WSS...")
wss = calculate_WSS(chromaticity, kmin, kmax)
elbow_k, distances = find_elbow_point(wss, kmin, kmax)

print(f"The optimal number of clusters based on the elbow method is: {elbow_k}")

# x = np.arange(kmin, kmax+1)
# y = np.array(wss)
# plot_elbow(x, y, elbow_k, kmin)

## Train cluster
# Clustering with optimal number of k from elbow
kmeans = KMeans(n_clusters=elbow_k, random_state=42).fit(chromaticity)
labels = kmeans.labels_.reshape(r.shape)

# Plot the chromaticity plane with clusters
plot_clusters(chromaticity, kmeans)

# Find clusters with most red (to our eyes)
red_cluster_idx = np.argmin(np.sum(kmeans.cluster_centers_, axis=1))

predictions = kmeans.predict(chromaticity)
img_labels = predictions.reshape(r.shape) 

plt.figure(figsize=(8, 6), dpi=80)
plt.imshow(img_labels)
plt.axis('off')
plt.show()

## Binary redbox video processing using cluster
redbox_cluster_mask = []

for frame in images:
    # print("Processing frame: ", len(redbox_cluster_mask) + 1)
    # Aplicar desfoque
    img_blur = cv2.GaussianBlur(frame, (7, 7), 2.0)
    img_blur = frame

    # Chromatic coordinates converter
    Y = img_blur.sum(axis=2)  # R + G + B
    Y[Y == 0] = 1e-6  # Substituir 0 por um valor muito pequeno
    
    # NaNs handling
    r = np.nan_to_num(img_blur[:, :, 0] / Y, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.nan_to_num(img_blur[:, :, 1] / Y, nan=0.0, posinf=0.0, neginf=0.0)
    
    chromaticity = np.float32(np.stack([r.flatten(), g.flatten()], axis=1))
    
    # Predictions using pre-trained kmeans
    predictions = kmeans.predict(chromaticity)
    labels = predictions.reshape(r.shape)
    
    # Masking
    redbox = (labels == red_cluster_idx)
    redbox_cluster_mask.append(redbox)

create_video("masks red", redbox_cluster_mask)

## Movement dyadic operation
movement_masks = []
prev_mask = redbox_cluster_mask[0]

for current_mask in redbox_cluster_mask[1:]:
    # Subtracting consecutive frames
    movement = cv2.absdiff(prev_mask.astype(np.uint8), current_mask.astype(np.uint8))
    movement_masks.append(movement)
    
    prev_mask = current_mask # update

create_video("movement", movement_masks)

## Moments
# Encontrar o centro de massa da região vermelha filtrada
moments_frames = []

for idx, redbox_frame in enumerate(redbox_cluster_mask):
    frame_original = images[idx]
    red_mask_uint8 = (redbox_frame * 255).astype(np.uint8)

    labeled, num_features = label(redbox_frame)
    centroids = center_of_mass(redbox_frame, labeled, np.arange(1, num_features + 1))

    # Image moments
    M = cv2.moments(red_mask_uint8)

    # a) Area
    area = M['m00']

    if area == 0:
        print("Área da máscara é zero. Nenhum objeto detectado.")
    else:
        # b) Centroids
        cX = M["m10"] / area
        cY = M["m01"] / area
        centroid = (int(cX), int(cY))

        # c) Central moments and inertia matrix
        mu20 = M["mu20"]
        mu02 = M["mu02"]
        mu11 = M["mu11"]
        inertia_matrix = np.array([[mu20, mu11], [mu11, mu02]])

        # d) Equivalent ellipse
        eigenvalues, eigenvectors = np.linalg.eig(inertia_matrix)

        a = 4 * np.sqrt(eigenvalues[0] / area)  # major axis length
        b = 4 * np.sqrt(eigenvalues[1] / area)  # minor axis length
        axes = (int(a / 2), int(b / 2))

        # e) Orientation
        theta_rad = 0.5 * np.arctan((2 * mu11) / (mu20 - mu02))
        theta_dgr = np.degrees(theta_rad)

        # f) Ellipse to Plot
        ellipse = cv2.ellipse(
            frame_original,
            centroid,
            axes,
            theta_dgr,
            0,
            360,
            (255, 0, 0),
            2,
        )

        cv2.circle(frame_original, centroid, 5, (0, 0, 255), -1)  # Centróide: ponto vermelho
        cv2.putText(frame_original, f"Angle: {theta_dgr:.2f} degrees", (centroid[0] - 50, centroid[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    moments_frames.append(frame_original)

create_video("processed_video", moments_frames)

# Closes all the frames
cv2.destroyAllWindows()