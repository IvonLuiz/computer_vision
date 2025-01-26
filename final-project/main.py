import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass
from scipy.ndimage import label

from sklearn.metrics import silhouette_score

# from utils import *

def subtract_frames(prev_frame, current_frame):
    diff = cv2.absdiff(prev_frame, current_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
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
prev_frame_blur = None

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:
        if prev_frame_blur is not None:

            frame_blur = cv2.GaussianBlur(frame, (5, 5), 1.5)
            movement_mask = subtract_frames(prev_frame_blur, frame_blur)
            movement.append(movement_mask)
            # cv2.imshow('frame', movement_mask)

            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        prev_frame_blur = frame
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

ref_img = images[1000]
ref_movement = movement[1000]

def rgb_splitter(image):
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17,7), sharey = True)
    for i in range(3):
        ax[i].imshow(image[:,:,i], cmap = rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize = 22)
        ax[i].axis('off')
    fig.tight_layout()


#rgb_splitter(ref_img)

img_blur = cv2.GaussianBlur(ref_img, (5, 5), 1.5)

Y = img_blur.sum(axis=2) # R + G + B
r = img_blur[:,:,0] / Y
g = img_blur[:,:,1] / Y

one_matrix = np.ones_like(float, shape=r.shape)
b = one_matrix- (r + g)

img_chromatic_rgb = cv2.merge([np.uint8(r*255),
                               np.uint8(g*255),
                               np.uint8(b*255)]) 
#cv2.imshow("Chromatic RGB img", img_chromatic_rgb)

def RG_Chroma_plotter(red, green):
    p_color = [(r, g, 1-r-g) for r,g in zip(red.flatten(), green.flatten())]
    
    norm = colors.Normalize(vmin=0,vmax=1.)
    norm.autoscale(p_color)
    p_color = norm(p_color).tolist()
    
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(red.flatten(), 
                green.flatten(), 
                c = p_color, alpha = 0.40)
    ax.set_xlabel('Red Channel', fontsize = 20)
    ax.set_ylabel('Green Channel', fontsize = 20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()

#RG_Chroma_plotter(r, g)

# Flatten the chromaticity coordinates for clustering
chromaticity = np.float32(np.stack([r.flatten(), g.flatten()], axis=1))


## Finding best k to cluster
def calculate_WSS(points, kmin, kmax):
  sse = []
  for k in range(kmin, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
    #sse.append(kmeans.inertia_)

  return sse

def find_elbow_point(wss, kmin, kmax):
    # Points for graph clusters
    x = np.arange(kmin, kmax+1)
    y = np.array(wss)
    
    # Defines line between first and last point
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    distances = []
    for i in range(len(x)):
        # Calculates perpendicular distance from each point to line
        p = np.array([x[i], y[i]])
        distance = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        distances.append(distance)
    
    # Best index
    elbow_index = np.argmax(distances)

    return x[elbow_index], distances

kmin = 2
kmax = 15
print("Calculating WSS...")
# wss = calculate_WSS(chromaticity, kmin, kmax)
# elbow_k, distances = find_elbow_point(wss, kmin, kmax)

# x = np.arange(kmin, kmax+1)
# y = np.array(wss)

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, marker='o', linestyle='-', label='WSS')
# plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
# plt.scatter([elbow_k], [y[elbow_k - kmin]], color='red', label='Elbow Point', zorder=5)

# plt.title('Elbow Method for Optimal k', fontsize=16)
# plt.xlabel('Number of Clusters (k)', fontsize=14)
# plt.ylabel('Within Sum of Squares (WSS)', fontsize=14)
# plt.xticks(x)
# plt.grid(alpha=0.3)
# plt.legend(fontsize=12)
# plt.show()
elbow_k = 5
print(f"The optimal number of clusters based on the elbow method is: {elbow_k}")


# Clustering with optimal number o k from elbow
kmeans = KMeans(n_clusters=elbow_k, random_state=42).fit(chromaticity)
labels = kmeans.labels_.reshape(r.shape)

# Plot the chromaticity plane with clusters
# plt.figure(figsize=(10, 6))
# plt.scatter(chromaticity[:, 0], chromaticity[:, 1], c=kmeans.labels_, s=1, alpha=0.35, cmap='tab10')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", marker="o", s=50, label="Centroids")
# plt.xlabel("Chromaticity r")
# plt.ylabel("Chromaticity g")
# plt.title("Chromaticity Plane with K-means Clusters")
# plt.legend()
#plt.show()

predictions = kmeans.predict(chromaticity)
img_labels = predictions.reshape(r.shape) 

# plt.figure(figsize=(8, 6), dpi=80)
# plt.imshow(img_labels)
# plt.axis('off')
#plt.show()

# Find cluster with red
red_cluster_idx = np.argmin(kmeans.cluster_centers_[:, 1])  # Maior valor de r
print(kmeans.cluster_centers_)

print(red_cluster_idx)
red_mask = (labels == red_cluster_idx)

# plt.figure(figsize=(8, 6), dpi=80)
# plt.imshow(red_mask, cmap='Reds')
# plt.title("Máscara do cluster vermelho")
# plt.axis('off')
#plt.show()

# Filtra a região de movimento
contours, _ = cv2.findContours(ref_movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Plot contours
plt.figure(figsize=(8, 6))
plt.imshow(ref_img)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
plt.title("Contours on Reference Image")
plt.axis('off')
plt.show()

# Create a mask for the moving region
movement_mask = np.zeros_like(ref_movement, dtype=np.uint8)

for contour in contours:
    print(contour)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(movement_mask, (x, y), (x + w, y + h), 255, -1)

# Filter red_mask if it is inside the contours of movement
filtered_red_mask = cv2.bitwise_and(red_mask.astype(np.uint8), red_mask.astype(np.uint8), mask=movement_mask)

plt.figure(figsize=(8, 6))
plt.imshow(filtered_red_mask, cmap='Reds')
plt.title("Filtered Red Mask")
plt.axis('off')
plt.show()
print("filtered red mask=", filtered_red_mask)

# Encontrar o centro de massa da região vermelha filtrada

labeled, num_features = label(filtered_red_mask)
centroids = center_of_mass(filtered_red_mask, labeled, np.arange(1, num_features + 1))

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