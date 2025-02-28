import numpy as np
import cv2
import matplotlib.pyplot as plt

filename = r"Images/Image6.jpg"

# Read the image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Apply Gaussian Blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply sharpening filter
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gray = cv2.filter2D(gray, -1, kernel)

# Detect corners using Harris Corner Detection with adjusted parameters
dst = cv2.cornerHarris(gray, 3, 5, 0.06)
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image
threshold = 0.01 * dst.max()
img[dst > threshold] = [0, 0, 255]
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
