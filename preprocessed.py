from PIL import Image
import os

# Paths
images_path = "home/rgukt-basar/bloodcell/images/JPEGImages"
output_path = "home/rgukt-basar/bloodcell/dataset/preprocessed"
os.makedirs(output_path, exist_ok=True)

# Resize each image
for img_file in os.listdir(images_path):
    img = Image.open(os.path.join(images_path, img_file)).convert('RGB')
    img = img.resize((256, 256))  # Resize to 256x256
    img.save(os.path.join(output_path, img_file))

