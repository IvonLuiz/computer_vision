import cv2
import os
import matplotlib.pyplot as plt

# Reading image
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "flowers9.png")
img = cv2.imread(img_path)

# Displaying original image
cv2.imshow("Image colored", img)
cv2.waitKey(0)


# Splitting into RGB channels
(B, G, R) = cv2.split(img)


# Displaying grayscale RGB images

# cv2.imshow('blue channel', B) 
# cv2.imshow('green channel', G) 
# cv2.imshow('red channel', R)

plt.subplot(1, 3, 1)
plt.imshow(R, cmap="gray")
plt.title("Red channel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(G, cmap="gray")
plt.title("Green channel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(B, cmap="gray")
plt.title("Blue channel")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'rgb_plots.png'))
plt.show()


# Hist plot from each channel

def hist_plot(channel_pixels, plot_position, channel_name=""):
    # Flatten the 2-D arrays of the RGB channels into 1-D
    channel_pixels_flat = channel_pixels.flatten()
    plt.subplot(*plot_position)
    plt.hist(channel_pixels_flat, bins=256, color='black', alpha=0.7)
    plt.title(f"Histogram from {channel_name} channel")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

plt.figure(figsize=(18, 6))
hist_plot(R, (1,3,1), "red")
hist_plot(G, (1,3,2), "green")
hist_plot(B, (1,3,3), "blue")

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'histograms.png'))
plt.show()

# Color planes
imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## Original RGB
plt.subplot(2, 2, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem RGB Original")
plt.axis("off")

## R channel
plt.subplot(2, 2, 2)
plt.imshow(R, cmap="Reds")
plt.title("Red channel")
plt.axis("off")

## G channel
plt.subplot(2, 2, 3)
plt.imshow(G, cmap="Greens")
plt.title("Green channel")
plt.axis("off")

## B channel
plt.subplot(2, 2, 4)
plt.imshow(B, cmap="Blues")
plt.title("Blue channel")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'plane_channels.png'))
plt.show()
