import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass
from scipy.ndimage import label

from utils import *

class Tracker:
    def __init__(self):
        self.kmeans = None
        self.red_cluster_idx = None
        self.frame_shape = None
    

    def train_kmeans(self, reference_frame, kmin, kmax):
        """
        Trains a KMeans clustering model to segment the chromaticity of an image using a reference frame.
        The method calculates the chromaticity of the input image, determines the optimal number of clusters
        using the Elbow method, and applies the KMeans algorithm to segment the image into clusters based on
        color information.
        """
        chromaticity = self.get_chromaticity(reference_frame)

        self.frame_shape = reference_frame.shape[:2]

        # Elbow method to determine optimal K
        wss = calculate_WSS(chromaticity, kmin, kmax)
        elbow_k, _ = find_elbow_point(wss, kmin, kmax)

        print(f"Optimal number of clusters: {elbow_k}")

        self.kmeans = KMeans(n_clusters=elbow_k, random_state=42).fit(chromaticity)
        self.red_cluster_idx = np.argmin(np.sum(self.kmeans.cluster_centers_, axis=1))
        
        plot_clusters(chromaticity, self.kmeans)


    def get_chromaticity(self, frame):
        """
        Calculates the chromaticity of an image by first applying a Gaussian blur to the input frame.
        Chromaticity represents the color intensity of the red and green channels relative to the 
        sum of all three color channels (Red, Green, and Blue).
        """

        frame_blur = cv2.GaussianBlur(frame, (7, 7), 2.0)
        Y = frame_blur.sum(axis=2) # R + G + B
        Y[Y == 0] = 1e-6

        r = np.nan_to_num(frame_blur[:, :, 0] / Y, nan=0.0, posinf=0.0, neginf=0.0)
        g = np.nan_to_num(frame_blur[:, :, 1] / Y, nan=0.0, posinf=0.0, neginf=0.0)

        chromaticity = np.float32(np.stack([r.flatten(), g.flatten()], axis=1))

        return chromaticity


    def process_video(self, video_path):
        """
        Processes a video to generate masks based on the trained KMeans model, tracks the red color cluster, 
        and calculates moments (such as centroid, area, and orientation) for each frame.
        The method creates two output videos:
        1. A video with the red cluster masked (where the red cluster is highlighted).
        2. A processed video showing the trajectory of the centroids and an ellipse representing the object's 
        orientation and shape.
        """
        if self.kmeans is None:
            raise ValueError("KMeans model has not been trained. Call train_kmeans() first.")

        cap = cv2.VideoCapture(video_path)
        redbox_cluster_mask = []
        moments_frames = []
        trajectory = []  # List to store centroid positions for trajectory
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Applying kmeans clustering from chromatic frame
            chromaticity = self.get_chromaticity(frame)

            predictions = self.kmeans.predict(chromaticity)
            labels = predictions.reshape(self.frame_shape)

            # Masking with red cluster
            redbox_frame = (labels == self.red_cluster_idx)
            redbox_cluster_mask.append(redbox_frame)


            ## Moments
            red_mask_uint8 = (redbox_frame * 255).astype(np.uint8)

            labeled, num_features = label(redbox_frame)
            centroids = center_of_mass(redbox_frame, labeled, np.arange(1, num_features + 1))

            M = cv2.moments(red_mask_uint8)
            
            # Area from object
            area = M['m00']
            
            if area > 0:
                # Centroids
                cX = M['m10'] / area
                cY = M['m01'] / area
                centroid = (int(cX), int(cY))

                # Saving trajectory with centroids and plotting
                trajectory.append(centroid)

                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)
                
                # Central moments and inertia matrix
                mu20, mu02, mu11 = M['mu20'], M['mu02'], M['mu11']
                inertia_matrix = np.array([[mu20, mu11], [mu11, mu02]])
                
                # Equivalent elipse
                eigenvalues, _ = np.linalg.eig(inertia_matrix)

                a = 4 * np.sqrt(eigenvalues[0] / area)
                b = 4 * np.sqrt(eigenvalues[1] / area)
                axes = (int(a / 2), int(b / 2))

                # Orientation
                theta_rad = 0.5 * np.arctan((2 * mu11) / (mu20 - mu02))
                theta_dgr = np.degrees(theta_rad)

                # 
                cv2.ellipse(
                    frame, centroid, axes, theta_dgr, 0, 360, (255, 0, 0), 2
                )
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"Angle: {theta_dgr:.2f}", (centroid[0] - 50, centroid[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

            moments_frames.append(frame)
        
        cap.release()

        create_video(f"{video_path}_redbox", redbox_cluster_mask)
        create_video(f"{video_path}_processed", moments_frames)