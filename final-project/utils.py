import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.cluster import KMeans
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


# -- Finding best k to cluster --

def calculate_WSS(points, kmin, kmax):
  sse = []
  for k in range(kmin, kmax+1):
    kmeans = KMeans(n_clusters = k, random_state=42).fit(points)
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

# -- Creating videos to save --

def create_video(path, frames):
    path = path + ".avi"
    fps = 30

    # Shape of video based on the first frame
    first_frame = frames[0]
    if len(first_frame.shape) == 2:
        # Grayscale frame
        frame_height, frame_width = first_frame.shape
        is_color = False
    elif len(first_frame.shape) == 3 and first_frame.shape[2] == 3:
        # RGB frame
        frame_height, frame_width, _ = first_frame.shape
        is_color = True
    else:
        print("Error: Invalid frame format. Frames should be 2D (grayscale) or 3D (RGB).")
        return

    # Initialize the writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    out = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height), isColor=is_color)

    # Create the video from frames
    print("Creating video...")

    for idx, frame in enumerate(frames):
        if idx%500 == 0:
            print(f"Processed {idx}/{len(frames)} frames")
        if is_color:
            frame = np.uint8(frame)  # Ensure the frame is in the correct format
        else:
            frame = np.uint8(frame * 255)  # Scale grayscale values to 0-255

        out.write(frame)

    # Release the writer and finish
    out.release()
    print(f"Video saved on {path}")


## -- Plots --

def rgb_splitter(image):
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17,7), sharey = True)
    for i in range(3):
        ax[i].imshow(image[:,:,i], cmap = rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize = 22)
        ax[i].axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(dir_path, "plots/rgb_splitter.png"))

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
    plt.savefig(os.path.join(dir_path, "plots/RG_Chroma.png"))

def plot_elbow(x, y, elbow_k, kmin):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', label='WSS')
    plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
    plt.scatter([elbow_k], [y[elbow_k - kmin]], color='red', label='Elbow Point', zorder=5)

    plt.title('Elbow Method for Optimal k', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Within Sum of Squares (WSS)', fontsize=14)
    plt.xticks(x)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(dir_path, "plots/elbow.png"))

def plot_clusters(chromaticity, kmeans):
    plt.figure(figsize=(10, 6))
    plt.scatter(chromaticity[:, 0], chromaticity[:, 1], c=kmeans.labels_, s=1, alpha=0.35, cmap='tab10')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", marker="o", s=50, label="Centroids")

    plt.xlabel("Chromaticity r")
    plt.ylabel("Chromaticity g")
    plt.title("Chromaticity Plane with K-means Clusters")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "plots/clusters.png"))

def plot_cluster_labels(kmeans, chromaticity, shape):
    predictions = kmeans.predict(chromaticity)
    img_labels = predictions.reshape(shape)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.imshow(img_labels)
    plt.axis('off')
    plt.savefig(os.path.join(dir_path, "plots/img_labels.png"))

def save_img_cv2(img, name):
    cv2.imwrite(os.path.join(dir_path, f"plots/{name}.png"), img)
