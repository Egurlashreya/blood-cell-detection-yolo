import os
import pandas as pd
from PIL import Image

csv_path = "../dataset/labels.csv"
images_path = "../images/JPEGImages"
output_root = "../dataset/cells"

df = pd.read_csv(csv_path)

# Create folders for all classes automatically
classes = df["class"].unique()
for c in classes:
    os.makedirs(os.path.join(output_root, c), exist_ok=True)

for index, row in df.iterrows():
    filename = row["filename"]
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
    cell_class = row["class"]

    image_path = os.path.join(images_path, filename)

    try:
        img = Image.open(image_path)
        w, h = img.size

        # Fix bounding box
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        if xmax <= xmin or ymax <= ymin:
            print(f"Skipping invalid box in {filename}")
            continue

        cropped = img.crop((xmin, ymin, xmax, ymax))

        save_path = os.path.join(output_root, cell_class, f"{index}_{filename}")
        cropped.save(save_path)

    except Exception as e:
        print(f"Error cropping {filename}: {e}")

print("Cropping completed!")

