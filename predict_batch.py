import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "bloodcell_model.h5"
IMAGE_FOLDER = "../dataset/test_images"     # folder where all images are placed

IMG_SIZE = (64, 64)              # use the same size used during training
CLASS_NAMES = ["rbc", "wbc", "platelets"]

model = load_model(MODEL_PATH)

def predict_single(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_id], confidence

print("Running batch prediction...\n")

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(IMAGE_FOLDER, filename)

        label, conf = predict_single(path)
        print(f"{filename} → {label} ({conf:.2f})")

