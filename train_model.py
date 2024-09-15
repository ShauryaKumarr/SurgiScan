import os
from ultralytics import YOLO

# Get the absolute path to your data.yaml file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'Utensil-Detection-1', 'data.yaml')

# Load the model
model = YOLO('yolov5su.pt')

# Train the model
results = model.train(data=data_yaml_path, epochs=100, imgsz=640)

# Evaluate the model
results = model.val()