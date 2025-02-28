import numpy as np
import cv2
import matplotlib.pyplot as plt

filename = r"Images/Image1.jpg"

# Read the image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Detect corners using Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image
img[dst > 0.01 * dst.max()] = [0, 0, 255]
img_rgb_corners = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the original and corner-detected images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_rgb_corners)
plt.title("Harris Corner Detected Image")
plt.axis("off")

plt.show()
