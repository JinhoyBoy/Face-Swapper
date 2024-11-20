import cv2
import numpy as np
import matplotlib.pyplot as plt

# Manually implemented resize function
def resize_image(image, new_size):
    height, width = new_size
    new_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    y_ratio = image.shape[0] / height
    x_ratio = image.shape[1] / width

    for i in range(height):
        for j in range(width):
            orig_y = int(i * y_ratio)
            orig_x = int(j * x_ratio)
            new_image[i, j] = image[orig_y, orig_x]

    return new_image

# Load an image using OpenCV
image = cv2.imread('application_images/1.jpg')  # Replace 'image.jpg' with your image path

# Define the new size for the image
new_size = (200, 300)  # Example: resizing to 200x300 pixels

# Resize using the manual function
manual_resized_image = resize_image(image, new_size)

# Resize using OpenCV's built-in resize function for comparison
opencv_resized_image = cv2.resize(image, (new_size[1], new_size[0]))

# Plot the original, manual resized, and OpenCV resized images for comparison
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(manual_resized_image, cv2.COLOR_BGR2RGB))
plt.title('Manually Resized Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(opencv_resized_image, cv2.COLOR_BGR2RGB))
plt.title('OpenCV Resized Image')
plt.axis('off')

plt.tight_layout()
plt.show()
