import os
import shutil
from sklearn.model_selection import train_test_split

input_dir = "../dataset/cells"
output_dir = "../dataset"

splits = ["train", "valid", "test"]
classes = os.listdir(input_dir)

# Create output folders
for split in splits:
    for c in classes:
        os.makedirs(os.path.join(output_dir, split, c), exist_ok=True)

# Split data class-wise
for c in classes:
    class_dir = os.path.join(input_dir, c)
    images = os.listdir(class_dir)

    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    valid_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

    # Move files
    for split, split_imgs in zip(["train", "valid", "test"],
                                 [train_imgs, valid_imgs, test_imgs]):
        for img in split_imgs:
            src = os.path.join(class_dir, img)
            dst = os.path.join(output_dir, split, c, img)
            shutil.copy(src, dst)

print("Dataset split completed!")

