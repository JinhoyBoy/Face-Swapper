import cv2
import numpy as np
import image_modules

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


# Face extraction and swapping (same as original code)
image1_name = input("input image1 name and format: ")
image2_name = input("input image2 name and format: ")

# Load the images
image_path1 = 'application_images/' + image1_name
image_path2 = 'application_images/' + image2_name

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect faces (same method as before)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5)
faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5)

if len(faces1) < 1 or len(faces2) < 1:
    print("There has to be at least two faces in the image.")
else:
    # Extract faces and resize
    x1, y1, w1, h1 = faces1[0]
    x2, y2, w2, h2 = faces2[0]
    face1 = image1[y1:y1 + h1, x1:x1 + w1]
    face2 = image2[y2:y2 + h2, x2:x2 + w2]
    
    # Resize the faces
    face1_resized = image_modules.resize_image(face1, (w2, h2))
    face2_resized = image_modules.resize_image(face2, (w1, h1))

    # Define the center and axes of the ellipses
    center1_resized = (w2 // 2, h2 // 2)
    axes1_resized = (w2 // 3, h2 // 2)
    center2_resized = (w1 // 2, h1 // 2)
    axes2_resized = (w1 // 3, h1 // 2)

    # Extract ellipses with gradient
    face1_ellipse = extract_ellipse_with_gradient(face1_resized, center1_resized, axes1_resized)
    face2_ellipse = extract_ellipse_with_gradient(face2_resized, center2_resized, axes2_resized)

    # Create result images
    result1 = image1.copy()
    result2 = image2.copy()

    # 2. Extract BGR channels and alpha channels from face1_ellipse and face2_ellipse
    face1_bgr = face1_ellipse[:, :, :3]  # BGR channels
    face1_alpha = face1_ellipse[:, :, 3]  # Alpha channel
    face2_bgr = face2_ellipse[:, :, :3]  # BGR channels
    face2_alpha = face2_ellipse[:, :, 3]  # Alpha channel

    # 3. Composite the alpha channels into result2's alpha channel
    # Composite only the parts where the alpha channel is not 0
    for i in range(h2):
        for j in range(w2):
            if face1_alpha[i, j] != 0:  # Only composite where the alpha channel is not 0
                # Blend the pixel based on the alpha value
                blend_factor = face1_alpha[i, j] / 255.0
                pixel = result2[y2 + i, x2 + j, :3] * (1 - blend_factor) + face1_bgr[i, j] * blend_factor
                pixel = np.clip(pixel, 0, 255)  # Clip the pixel values to be between 0 and 255
                result2[y2 + i, x2 + j, :3] = pixel

    # Same process for face2_ellipse to result1
    for i in range(h1):
        for j in range(w1):
            if face2_alpha[i, j] != 0:  # Only composite where the alpha channel is not 0
                # Blend the pixel based on the alpha value
                blend_factor = face2_alpha[i, j] / 255.0
                pixel = result1[y1 + i, x1 + j, :3] * (1 - blend_factor) + face2_bgr[i, j] * blend_factor
                pixel = np.clip(pixel, 0, 255)  # Clip the pixel values to be between 0 and 255
                result1[y1 + i, x1 + j, :3] = pixel

    # Show the result images
    cv2.imshow("Face Swap with Gradient Ellipses 1", result1)
    cv2.imshow("Face Swap with Gradient Ellipses 2", result2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
