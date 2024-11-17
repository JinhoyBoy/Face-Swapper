import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks, applications, Input

IMAGE_SIZE = 224

# Function to load images and labels
def load_images_and_labels(image_folder, labels_folder, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load label (normalized bounding box coordinates)
            label_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(labels_folder, label_filename)
            
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    box = values[1:]  # Ignore the first value (class label), use bounding box (x, y, width, height)
                    boxes.append(box)

            # Skip images with multiple bounding boxes
            if len(boxes) != 1:
                continue  # Skip images with no or multiple bounding boxes
            
            # Load and resize image only if it has a single bounding box
            img = cv2.imread(os.path.join(image_folder, filename))
            if img is None:  # Skip invalid images
                continue
            img = cv2.resize(img, target_size)  # Resize image to target size
            img = img / 255.0  # Normalize (pixel values between 0 and 1)
            
            images.append(img)
            labels.append(boxes[0])  # Append the single bounding box for the image

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


# Load validation images and labels
#val_images, val_labels = load_images_and_labels("valid/images", "valid/labels")
train_images, train_labels = load_images_and_labels("train_full/images", "train_full/labels")
print("Images and labels are loaded")

# Split data into training and validation sets (20% for validation)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Function to create the face detection model
def create_face_detection_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    inputs = Input(shape=input_shape)
    
    # Convolutional Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Convolutional Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Convolutional Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Convolutional Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(4)(x)  # 4 outputs for (x, y, width, height)
    
    # Model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

'''
def create_transfer_learning_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    # Load the MobileNetV2 model without the top layer
    base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    # Add custom layers for bounding box regression
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)  # Global pooling for feature reduction
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4)(x)  # Output layer for (x, y, width, height)

    # Create the model
    model = models.Model(inputs, outputs)
    return model
'''

# Create the model
model = create_face_detection_model()
#model = create_transfer_learning_model()
model.summary()


# Loss, optimizer, metrics
#model.compile(optimizer='adam', loss=keras_cv.losses.IoULoss(bounding_box_format="xyWH"), metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print("Training data length:", len(train_images))
print("Validation data length:", len(val_images))

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels= np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Early stopping callback to stop training early if the validation loss stops decreasing and save the best model
#early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    callbacks.ModelCheckpoint('trained_model.keras', save_best_only=True, monitor='val_loss')
]

# Train model with simplified single-bounding-box data
history = model.fit(
    train_images, train_labels,
    epochs=100,
    batch_size=32,
    validation_data=(val_images, val_labels),
    callbacks=callbacks,
    verbose=1
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss & Acuuracy')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Save the model
print("Model saved as trained_model.keras")

# Load validation images and labels
test_images, test_labels = load_images_and_labels("test/images", "test/labels")
print("Test images and labels are loaded")
loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)