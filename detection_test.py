import keras
import cv2
import numpy as np

IMAGE_SIZE = 360

# Load the trained model
model = keras.models.load_model("face_model.keras")

# Function to preprocess the image
def preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img = cv2.resize(image, target_size)  # Resize image to target size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to display bounding box on the resized image
def display_bounding_box(image, bounding_box):
    x, y, w, h = bounding_box

    # Calculate the bounding box coordinates on the resized image
    x_min = int(x * IMAGE_SIZE - w * IMAGE_SIZE / 2)
    y_min = int(y * IMAGE_SIZE - h * IMAGE_SIZE / 2)
    x_max = int(x * IMAGE_SIZE + w * IMAGE_SIZE / 2)
    y_max = int(y * IMAGE_SIZE + h * IMAGE_SIZE / 2)

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

# List of image paths
image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]

# Process each image
for image_path in image_paths:
    image_path = "application_images/" + image_path
    # Load a new image for face detection
    original_image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if original_image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Resize the image for processing and display
    resized_image = cv2.resize(original_image, (IMAGE_SIZE, IMAGE_SIZE))

    # Preprocess the resized image
    input_image = preprocess_image(resized_image)

    # Predict bounding box
    predicted_box = model.predict(input_image)[0]  # Get the first (and only) prediction
    print(f"Predicted bounding box for {image_path}:", predicted_box)

    # Display the bounding box directly on the resized image
    output_image = display_bounding_box(resized_image, predicted_box)

    # Show the image with the bounding box
    cv2.imshow("Detected Face", output_image)
    cv2.waitKey(0)  # Wait for a key press to show the next image

# Close all OpenCV windows
cv2.destroyAllWindows()
