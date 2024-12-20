import numpy as np
import math

# replaces cv2.merge
def merge_channels(channels):
    # Stack the individual channels along the last axis to create a multi-channel image (e.g., RGB or RGBA)
    return np.stack(channels, axis=-1)

# replaces cv2.resize
def resize_image(image, new_size):
    # Extract the new height and width from the new_size tuple
    height, width = new_size
    
    # Create a new empty image with the desired size and the same number of channels as the input image
    new_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    
    # Calculate the ratio of the original size to the new size for both dimensions
    y_ratio = image.shape[0] / height  # Vertical scaling factor
    x_ratio = image.shape[1] / width   # Horizontal scaling factor

    # Iterate through each pixel of the new image
    for i in range(height):
        for j in range(width):
            # Find the corresponding pixel in the original image
            orig_y = int(i * y_ratio)  # The y-coordinate of the original pixel
            orig_x = int(j * x_ratio)  # The x-coordinate of the original pixel
            
            # Assign the pixel value from the original image to the resized image
            new_image[i, j] = image[orig_y, orig_x]

    return new_image

# replaces cv2.bitwise_and
def bitwise_and(image1, image2):
    # Ensure both images are of the same size and number of channels
    if image1.shape != image2.shape:
        raise ValueError("The input images must have the same shape.")

    # Perform the bitwise AND operation pixel by pixel for all channels
    result = np.bitwise_and(image1, image2)

    return result

def draw_filled_ellipse(img, center, axes, angle, color):
    h, k = center  # Center of the ellipse
    a, b = axes  # Semi-major and semi-minor axes
    angle = math.radians(angle)  # Convert rotation angle to radians

    # Compute cosine and sine of the rotation angle once
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    # Define the bounding box of the ellipse
    min_x = max(0, int(h - a))
    max_x = min(img.shape[1], int(h + a))
    min_y = max(0, int(k - b))
    max_y = min(img.shape[0], int(k + b))

    # Loop over the bounding box and check if points are inside the ellipse
    for px in range(min_x, max_x):
        for py in range(min_y, max_y):
            # Translate the point to the ellipse's local coordinates
            x = px - h
            y = py - k

            # Apply the reverse rotation to the point
            x_rot = x * cos_angle + y * sin_angle
            y_rot = -x * sin_angle + y * cos_angle

            # Check if the point is inside the ellipse using the standard equation
            if (x_rot ** 2) / (a ** 2) + (y_rot ** 2) / (b ** 2) <= 1:
                img[py, px] = color
