import cv2
import numpy as np

# Load the image
img = cv2.imread("Lena.jpg", 0)

# Apply Gaussian blur for noise reduction
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Compute the gradients using the Sobel filter
grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

# Compute the magnitude and orientation of the gradient
magnitude = np.sqrt(grad_x**2 + grad_y**2)
orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi

# Apply non-maximum suppression
nms = cv2.Canny(img, 50, 150, L2gradient=True)

# Display the images
cv2.imshow('Original', img)
cv2.imshow('Gradient X', grad_x)
cv2.imshow('Gradient Y', grad_y)
cv2.imshow('Magnitude', magnitude)
cv2.imshow('Orientation', orientation)
cv2.imshow('Non-Max Suppression', nms)
cv2.waitKey(0)
cv2.destroyAllWindows()
