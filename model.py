import tensorflow as tf
from keras import layers, models
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load images and labels
def load_images_and_labels(image_folder, labels_folder, target_size=(416, 416)):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load and resize image
            img = cv2.imread(os.path.join(image_folder, filename))
            img = cv2.resize(img, target_size)  # Resize image to target size
            img = img / 255.0  # Normalize (pixel values between 0 and 1)
            images.append(img)
            
            # Load label (normalized bounding box coordinates)
            label_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(labels_folder, label_filename)
            
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    box = values[1:]  # Ignore the first value (class label), use bounding box (x, y, width, height)
                    boxes.append(box)
            labels.append(boxes)  # Append all bounding boxes for the image

    # Convert images to numpy array
    images = np.array(images)

    # Keep labels as a list of lists to handle varying number of bounding boxes per image
    return images, labels

# Load validation images and labels
val_images, val_labels = load_images_and_labels("train/images", "train/labels")
print("Images and labels are loaded")

# Split into training and validation sets (e.g., 80% training, 20% validation)
train_images, val_images, train_labels, val_labels = train_test_split(
    val_images, val_labels, test_size=0.2, random_state=42
)

# Define model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(416, 416, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)  # Output: x, y, width, height for a single bounding box
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training loop to handle images with a single bounding box (for simplicity)
train_images_single_box = [img for img, boxes in zip(train_images, train_labels) if len(boxes) == 1]
train_labels_single_box = [boxes[0] for boxes in train_labels if len(boxes) == 1]
val_images_single_box = [img for img, boxes in zip(val_images, val_labels) if len(boxes) == 1]
val_labels_single_box = [boxes[0] for boxes in val_labels if len(boxes) == 1]

# Convert lists to numpy arrays
train_images_single_box = np.array(train_images_single_box)
train_labels_single_box = np.array(train_labels_single_box)
val_images_single_box = np.array(val_images_single_box)
val_labels_single_box = np.array(val_labels_single_box)

# Train model with simplified single-bounding-box data
history = model.fit(
    train_images_single_box, train_labels_single_box,
    epochs=30,
    batch_size=32,
    validation_data=(val_images_single_box, val_labels_single_box)
)

# Save the model
model.save("face_detection_model.keras")
