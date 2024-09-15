# train YOLO using RoboFlow datasets
import os
from ultralytics import YOLO


current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'Utensil-Detection-2', 'data.yaml')


model = YOLO('yolov5su.pt')


results = model.train(data=data_yaml_path, epochs=100, imgsz=640)


results = model.val()