import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass
from scipy.ndimage import label

from sklearn.metrics import silhouette_score

from utils import *

def subtract_frames(prev_frame, current_frame):
    diff = cv2.absdiff(prev_frame, current_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 22, 255, cv2.THRESH_BINARY)
    return thresh

def draw_bounding_box(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return (x, y, w, h)
    return None

def get_clusters_in_box(clusters, box):
    x, y, w, h = box
    return clusters[y:y + h, x:x + w]


video_name = "Video1_husky.mp4"

dir_path = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(dir_path, video_name)

cap = cv2.VideoCapture(video_path)
images = []
movement = []
# prev_frame_blur = None

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:
        # if prev_frame_blur is not None:

            # frame_blur = cv2.GaussianBlur(frame, (7, 7), 2.5)
            # movement_mask = subtract_frames(prev_frame_blur, frame_blur)
            # movement.append(movement_mask)
            # cv2.imshow('frame', movement_mask)

            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # prev_frame_blur = frame
        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray_images.append(gray_image)
        
        # print("added image")

        images.append(frame)
    #     # Display the resulting frame
    #     cv2.imshow('Frame', frame)
            
    #     # Press Q on keyboard to exit
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break

    # Break the loop
    else:
        print("all images added")
        break

ref_img = images[670] # 670 moment that pass on a obstacle
# ref_movement = movement[500]


rgb_splitter(ref_img)

img_blur = ref_img
img_blur = cv2.GaussianBlur(ref_img, (5, 5), 1.0)

Y = img_blur.sum(axis=2) # R + G + B
r = img_blur[:,:,0] / Y
g = img_blur[:,:,1] / Y

one_matrix = np.ones_like(float, shape=r.shape)
b = one_matrix- (r + g)

img_chromatic_rgb = cv2.merge([np.uint8(r*255),
                               np.uint8(g*255),
                               np.uint8(b*255)]) 
cv2.imshow("Chromatic RGB img", img_chromatic_rgb)



# RG_Chroma_plotter(r, g)

# Flatten the chromaticity coordinates for clustering
chromaticity = np.float32(np.stack([r.flatten(), g.flatten()], axis=1))


## Search for optimal value of K clusters with elbow method
kmin = 2
kmax = 15

print("Calculating WSS...")

wss = calculate_WSS(chromaticity, kmin, kmax)
elbow_k, distances = find_elbow_point(wss, kmin, kmax)
# elbow_k = 5
print(f"The optimal number of clusters based on the elbow method is: {elbow_k}")

x = np.arange(kmin, kmax+1)
y = np.array(wss)
plot_elbow(x, y, elbow_k, kmin)

## Clustering
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


print(red_cluster_idx)
red_mask = (labels == red_cluster_idx)

plt.figure(figsize=(8, 6), dpi=80)
plt.imshow(red_mask, cmap='Reds')
plt.title("Máscara do cluster vermelho")
plt.axis('off')
plt.show()

# Filtra a região de movimento
# contours, _ = cv2.findContours(ref_movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Plot contours
# plt.figure(figsize=(8, 6))
# plt.imshow(ref_img)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
#     plt.gca().add_patch(rect)
# plt.title("Contours on Reference Image")
# plt.axis('off')
# plt.show()

# Create a mask for the moving region
# movement_mask = np.zeros_like(ref_movement, dtype=np.uint8)

# for contour in contours:
#     print(contour)
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(movement_mask, (x, y), (x + w, y + h), 255, -1)

# # Filter red_mask if it is inside the contours of movement
# filtered_red_mask = cv2.bitwise_and(red_mask.astype(np.uint8), red_mask.astype(np.uint8), mask=movement_mask)

# plt.figure(figsize=(8, 6))
# plt.imshow(filtered_red_mask, cmap='Reds')
# plt.title("Filtered Red Mask")
# plt.axis('off')
# plt.show()
# print("filtered red mask=", filtered_red_mask)

# # Encontrar o centro de massa da região vermelha filtrada

labeled, num_features = label(red_mask)
centroids = center_of_mass(red_mask, labeled, np.arange(1, num_features + 1))

# Plotar os centroides na máscara
plt.figure(figsize=(8, 6))
#plt.imshow(red_mask, cmap='Reds')

for c in centroids:
    plt.scatter(c[1], c[0], color='blue', marker='x', s=100, label='Centroide')
plt.title("Centroides na máscara do cluster vermelho")
plt.legend()
plt.axis('off')
#plt.show()

# Salvando os centroides para cada frame
pose_data = []
pose_data.append(centroids)

print(red_mask)
# Converting to binary
#gray_image = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
#_, binary_img = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

red_mask_uint8 = (red_mask * 255).astype(np.uint8)
red_mask_uint8 = filtered_red_mask


# Image moments
M = cv2.moments(red_mask_uint8)

# a) Area
area = M['m00']
print(f"Area from object: {area}")

# b) Centroids
cX = M["m10"] / area
cY = M["m01"] / area
centroids = (int(cX), int(cY))
print("centroids", centroids)

cv2.circle(ref_img, centroids, 20, (255, 0, 0), 2)
plt.figure(figsize=(8, 6), dpi=80)
plt.imshow(ref_img)
plt.axis('off')
plt.show()

# for img in images:
    ## Converting to binary
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary_img = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    
    # # Image moments
    # M = cv2.moments(binary_img)

    # # a) Area
    # area = M['m00']
    # print(f"Area from object: {area}")

    # # b) Centroids
    # cX = M["m10"] / area
    # cY = M["m01"] / area
    # centroids = (int(cX), int(cY))
    # cv2.imshow('Framee', img)
    # # Press Q on keyboard to exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break



# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()