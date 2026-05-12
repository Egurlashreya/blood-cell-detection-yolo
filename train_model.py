import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -----------------------------
# Paths
# -----------------------------
train_dir = "../dataset/train"
valid_dir = "../dataset/valid"
test_dir  = "../dataset/test"

# -----------------------------
# Hyperparameters
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4  # Update to match your dataset folders

# -----------------------------
# Data generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen  = ImageDataGenerator(rescale=1/255.0)

# Flow images from folders
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# Load MobileNetV2 base model
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze feature extractor

# -----------------------------
# Build full model
# -----------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')  # Match number of classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train the model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS
)

# -----------------------------
# Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)

# -----------------------------
# Save the trained model
# -----------------------------
model.save("../blood_cell_classifier.h5")
print("Model training completed! Saved as blood_cell_classifier.h5")

