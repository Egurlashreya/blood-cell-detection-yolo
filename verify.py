import os
from PIL import Image

# Paths to dataset splits
folders = [
    "../dataset/train",
    "../dataset/valid",
    "../dataset/test"
]

# Function to check and delete corrupted images
def remove_corrupted_images(base_path):
    removed_count = 0
    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify the image integrity
            except (IOError, SyntaxError):
                print(f"Removing corrupted image: {img_path}")
                os.remove(img_path)
                removed_count += 1
    return removed_count

total_removed = 0
for folder in folders:
    total_removed += remove_corrupted_images(folder)

print(f"Finished! Total corrupted images removed: {total_removed}")

