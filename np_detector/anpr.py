
# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics

from ultralytics import YOLO
import numpy as np

# Constants
SAVED_MODEL = "../models/my_yolo11n_model(new).pt"

# dataset_directory_path = "../data/IndianNumberPlate-dataset"

def test_code():
    print("test code here")

""" Detect numberplate bounding box using yolo11n trained model """


# load saved model
try:
    trained_model = YOLO(SAVED_MODEL)
except Exception as e:
    print("Error loading model:", e)
    exit(1)

def detect_number_plate(image_path):
    results = trained_model.predict(image_path,conf=0.6)  # Returns a list
    for result in results:  # Loop through results (usually one item in the list)
        result.show()  # Show the detected image
        # Extract detailed predictions
    detections = []
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        cls_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        class_name = trained_model.names[cls_id]

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "box": [round(coord, 2) for coord in xyxy]
        })

    return detections

