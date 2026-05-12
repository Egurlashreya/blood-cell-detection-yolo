import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224

# Load model
model = tf.keras.models.load_model("../blood_cell_classifier.h5")

class_names = ["rbc", "wbc", "platelets"]   # ensure same order as folders

def predict_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    print(f"Prediction: {class_names[class_idx]}")
    print(f"Confidence: {confidence:.2f}")

# Test it
predict_image("sample.jpeg")   # replace with your image

