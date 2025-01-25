import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass
from scipy.ndimage import label


video_name = "Video1_husky.mp4"

dir_path = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(dir_path, video_name)

cap = cv2.VideoCapture(video_path)
images = []

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:


        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray_images.append(gray_image)
        
        # print("added image")

        images.append(frame)
    # if ret == True:
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

def rgb_splitter(image):
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17,7), sharey = True)
    for i in range(3):
        ax[i].imshow(image[:,:,i], cmap = rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize = 22)
        ax[i].axis('off')
    fig.tight_layout()


rgb_splitter(ref_img)

img_blur = cv2.GaussianBlur(ref_img, (5, 5), 1.5)

Y = img_blur.sum(axis=2) # R + G + B
r = img_blur[:,:,0] / Y
g = img_blur[:,:,1] / Y

one_matrix = np.ones_like(float, shape=r.shape)
b = one_matrix- (r + g)

img_chromatic_rgb = cv2.merge([np.uint8(r*255),
                               np.uint8(g*255),
                               np.uint8(b*255)]) 
cv2.imshow("Chromatic RGB img", img_chromatic_rgb)

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

RG_Chroma_plotter(r, g)

# Flatten the chromaticity coordinates for clustering
chromaticity = np.float32(np.stack([r.flatten(), g.flatten()], axis=1))

kmeans = KMeans(n_clusters=7, random_state=42).fit(chromaticity)
labels = kmeans.labels_.reshape(r.shape)
print(labels)

# Plot the chromaticity plane with clusters
plt.figure(figsize=(10, 6))
plt.scatter(chromaticity[:, 0], chromaticity[:, 1], c=kmeans.labels_, s=1, alpha=0.35, cmap='tab10')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", marker="o", s=50, label="Centroids")
plt.xlabel("Chromaticity r")
plt.ylabel("Chromaticity g")
plt.title("Chromaticity Plane with K-means Clusters")
plt.legend()
plt.show()

predictions = kmeans.predict(chromaticity)
img_labels = predictions.reshape(r.shape) 

plt.figure(figsize=(8, 6), dpi=80)
plt.imshow(img_labels)
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