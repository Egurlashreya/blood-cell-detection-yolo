import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# Paths
annotations_path = "/home/rgukt-basar/bloodcell/images/Annotations"
csv_output_path = "/home/rgukt-basar/bloodcell/dataset/labels.csv"

# Ensure dataset folder exists
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

# Get all XML files
all_files = glob.glob(os.path.join(annotations_path, "*.xml"))

xml_list = []

for xml_file in all_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find("filename").text
    
    for member in root.findall("object"):
        class_name = member.find("name").text
        bndbox = member.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        xml_list.append((filename, xmin, ymin, xmax, ymax, class_name))

# Create DataFrame
column_name = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
xml_df = pd.DataFrame(xml_list, columns=column_name)

# Save CSV
xml_df.to_csv(csv_output_path, index=False)
print(f"CSV saved to {csv_output_path}")

