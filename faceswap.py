import cv2
import keras
import numpy as np
import image_modules
import matplotlib.pyplot as plt

# Load the trained model
IMAGE_SIZE = 360
MODEL = keras.models.load_model("face_detection_model.keras")

# Function to preprocess the image
def preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img = image_modules.resize_image(image, target_size)  # Resize image to target size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make a bounding box from the predicted center coordinates
def make_bounding_box(bounding_box, original_shape):
    h_orig, w_orig = original_shape[:2]
    cx, cy, w, h = bounding_box  # Assume cx, cy are the center coordinates

    # Convert bounding box from normalized center coordinates to pixel coordinates
    cx = int(cx * w_orig)
    cy = int(cy * h_orig)
    w = int(w * w_orig)
    h = int(h * h_orig)

    # Determine the side length of the square
    side_length = max(w, h)

    # Calculate the top-left corner (x, y) for the square box
    x = int(cx - side_length // 2)
    y = int(cy - side_length // 2)

    # Ensure the square box does not go outside the image boundaries
    if x < 0:
        x = 0
    if x + side_length > w_orig:
        side_length = w_orig - x

    if y < 0:
        y = 0
    if y + side_length > h_orig:
        side_length = h_orig - y

    return np.array([x, y, side_length, side_length])


# Function to generate an ellipse gradient mask
def generate_ellipse_gradient_mask(image_shape, center, axes, angle=0):
    height, width = image_shape[:2]
    
    # Create an empty array to store the gradient mask (2D array)
    gradient_mask = np.zeros((height, width), dtype=np.float32)
    
    # Calculate the gradient based on the ellipse equation
    for y in range(height):
        for x in range(width):
            # Check if the point lies inside the ellipse
            dx = (x - center[0]) * np.cos(angle) + (y - center[1]) * np.sin(angle)
            dy = -(x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle)
            
            # Ellipse equation (x^2 / a^2 + y^2 / b^2 <= 1)
            ellipse_distance = (dx ** 2) / (axes[0] ** 2) + (dy ** 2) / (axes[1] ** 2)
            
            if ellipse_distance <= 1:
                # Inside the ellipse: Apply gradient (more transparent towards the edges)
                gradient_mask[y, x] = np.sqrt(1 - ellipse_distance)  # Value between 0 and 1
            else:
                gradient_mask[y, x] = 0  # Outside the ellipse is 0 (transparent)
    
    return gradient_mask

# Function to extract an ellipse with gradient effect
def extract_ellipse_with_gradient(image, center, axes, angle=0):
    # Create a gradient mask with the ellipse effect
    gradient_mask = generate_ellipse_gradient_mask(image.shape, center, axes, angle)
    
    # Apply the mask to the image (convert to BGRA image with alpha channel)
    if image.shape[2] == 3:
        # Add an alpha channel
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        image = np.dstack([image, alpha_channel])
    
    # gradient_mask should be in (height, width) format
    # Apply the gradient mask to the alpha channel (values should be scaled to [0, 255])
    image[..., 3] = (gradient_mask * 255).astype(np.uint8)  # Apply gradient to the alpha channel
    
    return image

def resize_image_with_aspect_ratio(image, max_width=None, max_height=None):
    # Get the original dimensions
    h, w = image.shape[:2]

    # Calculate the scaling factor
    if max_width is not None and max_height is not None:
        scale = min(max_width / w, max_height / h)
    elif max_width is not None:
        scale = max_width / w
    elif max_height is not None:
        scale = max_height / h
    else:
        raise ValueError("At least one of max_width or max_height must be specified.")

    # Ensure scale is <= 1 (only reduce size)
    scale = min(scale, 1.0)

    # Calculate new dimensions
    new_width = int(w * scale)
    new_height = int(h * scale)

    # Resize the image
    resized_image = image_modules.resize_image(image, (new_height, new_width))

    return resized_image

# Face extraction and swapping (same as original code)
image1_name = input("input source image (image1) name and format: ")
image2_name = input("input swap image (image2) name and format: ")

# Load the images
image_path1 = 'application_images/' + image1_name
image_path2 = 'application_images/' + image2_name

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

image2 = resize_image_with_aspect_ratio(image2, max_width=800)

if image1.shape != image2.shape:
    print("The images must have the same shape. resizing the first image to be the same shape.")
    image1 = image_modules.resize_image(image1, (image2.shape[0], image2.shape[1]))

# prediction
image1_resized = image_modules.resize_image(image1, (IMAGE_SIZE, IMAGE_SIZE)) # model input size is 360x360
image2_resized = image_modules.resize_image(image2, (IMAGE_SIZE, IMAGE_SIZE)) # model input size is 360x360

input_image1 = preprocess_image(image1_resized)
input_image2 = preprocess_image(image2_resized)

predicted_box1 = make_bounding_box(MODEL.predict(input_image1)[0], image2.shape)
predicted_box2 = make_bounding_box(MODEL.predict(input_image2)[0], image1.shape)

print("Source Image shape:", image1.shape)
print("Result Image shape:", image2.shape)
print("Predicted bounding box for image1:", predicted_box1)
print("Predicted bounding box for image2:", predicted_box2)

# Face swapping
if predicted_box1 is None or predicted_box2 is None:
    print("There has to be at least two faces in the image.")
else:
    # Extract faces and resize
    x1, y1, w1, h1 = predicted_box1
    x2, y2, w2, h2 = predicted_box2
    face1 = image1[y1:y1 + h1, x1:x1 + w1]
    
    # Resize the faces
    face1_resized = image_modules.resize_image(face1, (w2, h2))

    # Define the center and axes of the ellipses
    center1_resized = (w2 // 2, h2 // 2)
    axes1_resized = (w2 // 3, h2 // 2)

    # Extract ellipses with gradient
    face1_ellipse = extract_ellipse_with_gradient(face1_resized, center1_resized, axes1_resized)

    # Create result images
    result = image2.copy()

    # Extract BGR channels and alpha channels from face1_ellipse and face2_ellipse
    face1_bgr = face1_ellipse[:, :, :3]  # BGR channels
    face1_alpha = face1_ellipse[:, :, 3]  # Alpha channel

    # Composite the alpha channels into result2's alpha channel
    # Composite only the parts where the alpha channel is not 0
    for i in range(min(h2, result.shape[0] - y2)):
        for j in range(min(w2, result.shape[1] - x2)):
            if face1_alpha[i, j] != 0:
                blend_factor = face1_alpha[i, j] / 255.0
                pixel = result[y2 + i, x2 + j, :3] * (1 - blend_factor) + face1_bgr[i, j] * blend_factor
                pixel = np.clip(pixel, 0, 255)
                result[y2 + i, x2 + j, :3] = pixel

    # Display the images
    plt.figure(figsize=(10, 6), num="Face Swapping")

    plt.subplot(1, 3, 1)
    plt.imshow(image1[:, :, ::-1]) # Convert BGR to RGB
    plt.title('Source Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image2[:, :, ::-1]) # Convert BGR to RGB
    plt.title('Image to Swap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result[:, :, ::-1]) # Convert BGR to RGB
    plt.title('Result Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    
