# Blood Cell Detection using YOLO

A deep learning-based computer vision project that detects and classifies blood cells from microscopic blood sample images using the YOLO object detection model. The system identifies Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets through image preprocessing, model training, inference, and evaluation workflows.

---

## Project Overview

This project applies object detection and deep learning techniques in the healthcare domain to automate blood cell analysis from microscopic images. The model processes blood sample images, detects different blood cell types, and visualizes predictions using bounding boxes.

---

## Features

* Detection and classification of:

  * Red Blood Cells (RBC)
  * White Blood Cells (WBC)
  * Platelets
* Image preprocessing and annotation handling
* YOLO-based object detection pipeline
* Batch prediction support
* Bounding box visualization
* Model evaluation and inference workflows

---

## Tech Stack

* Python
* YOLO
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib

---

## Project Structure

```bash
bloodcell/
│
├── dataset/                 # Blood cell image dataset
├── images/                  # Sample images and outputs
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks
├── results/                 # Prediction outputs
├── src/                     # Source code files
│
├── train_model.py           # Model training script
├── predict.py               # Single image prediction
├── predict_batch.py         # Batch prediction script
├── preprocesssed.py         # Data preprocessing
├── cropped_cells.py         # Cell extraction utilities
├── split_data.py            # Dataset splitting
├── verify.py                # Data verification
├── xml_to_csv.py            # Annotation conversion
│
├── labels.csv               # Dataset labels
├── sample.jpeg              # Sample input image
├── test.jpg                 # Test image
├── blood_cell_classifier.h5 # Trained model
```

---

## Workflow

1. Collect and organize blood cell image datasets
2. Perform preprocessing and annotation handling
3. Train YOLO-based object detection model
4. Perform inference on test images
5. Evaluate prediction performance and visualize outputs

---

## Applications

* Medical image analysis
* Automated blood diagnostics
* AI-assisted healthcare systems
* Computer vision research

---

## Future Improvements

* Improve model accuracy using larger datasets
* Add real-time webcam detection
* Deploy as a web application
* Integrate advanced visualization dashboards

---

## Author

Shreya Egurla
B.Tech CSE Student | AI/ML Enthusiast
