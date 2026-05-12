import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Blood Cell Detection using YOLOv8")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Blood Cell Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image directly
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy array (NO temp file needed)
    results = model.predict(image)

    r = results[0]

    # Show output image
    st.image(r.plot(), caption="Detected Cells", use_container_width=True)

    # Count cells
    counts = {}
    for cls_id in r.boxes.cls:
        name = r.names[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1

    st.subheader("Cell Counts")

    for k, v in counts.items():
        st.write(f"{k}: {v}")
