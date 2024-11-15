import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("face_detection_model.keras")

# Function to preprocess the image
def preprocess_image(image, target_size=(416, 416)):
    img = cv2.resize(image, target_size)  # Resize image to target size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to display bounding box on the original image
def display_bounding_box(image, bounding_box):
    height, width, _ = image.shape
    x, y, w, h = bounding_box

    # Scale bounding box coordinates back to the original image size
    x_min = int((x - w / 2) * width)
    y_min = int((y - h / 2) * height)
    x_max = int((x + w / 2) * width)
    y_max = int((y + h / 2) * height)

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

# Load a new image for face detection
image_path = "test_images/person.jpeg"
image = cv2.imread(image_path)

# Preprocess the image
input_image = preprocess_image(image)

# Predict bounding box
predicted_box = model.predict(input_image)[0]  # Get the first (and only) prediction
print("Predicted bounding box:", predicted_box)

# Display the bounding box on the original image
output_image = display_bounding_box(image, predicted_box)

# Show the image with bounding box
cv2.imshow("Detected Face", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
