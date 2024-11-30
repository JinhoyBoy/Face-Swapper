import keras
import cv2
import os
import numpy as np

IMAGE_SIZE = 360
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Load the trained model
model = keras.models.load_model("face_detection_model.keras")

# Function to preprocess the image
def preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img = cv2.resize(image, target_size)  # Resize image to target size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to display bounding box on the resized image
def display_bounding_box(image, bounding_box):
    x, y, w, h = bounding_box

    print("x, y, w, h: ", x, y, w, h)

    # Calculate the bounding box coordinates on the resized image
    x_min = int((x - w / 2) * IMAGE_SIZE)
    y_min = int((y - h / 2) * IMAGE_SIZE)
    x_max = int((x + w / 2) * IMAGE_SIZE)
    y_max = int((y + h / 2) * IMAGE_SIZE)

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

# Specify the folder containing the images
image_folder = "application_images/"

# Check if folder exists
if not os.path.exists(image_folder):
    print(f"Error: Folder '{image_folder}' does not exist.")
    exit()

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

# Check if there are any images to process
if not image_files:
    print(f"No images found in the folder '{image_folder}'.")
    exit()

# Process each image in the folder
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Load the image for face detection
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
    try:
        predicted_box = model.predict(input_image)[0]  # Get the first (and only) prediction
        print(f"Predicted bounding box for {image_path}:", predicted_box)

        # Display the bounding box directly on the resized image
        output_image = display_bounding_box(resized_image, predicted_box)

        # Show the image with the bounding box
        cv2.imshow(f"{image_path.replace(image_folder, '')}", output_image)
        cv2.waitKey(0)  # Wait for a key press to show the next image

    except Exception as e:
        print(f"Error predicting bounding box for {image_path}: {e}")
        continue

# Close all OpenCV windows
cv2.destroyAllWindows()
