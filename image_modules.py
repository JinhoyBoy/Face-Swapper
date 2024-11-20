import numpy as np

def merge_channels(channels):
    # Stack the individual channels along the last axis to create a multi-channel image (e.g., RGB or RGBA)
    return np.stack(channels, axis=-1)

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
