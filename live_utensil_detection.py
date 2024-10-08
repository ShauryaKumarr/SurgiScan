import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('/opt/homebrew/runs/detect/train7/weights/best.pt')


cap = cv2.VideoCapture(0)

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to capture frame")

frame_height, frame_width = frame.shape[:2]

# Define the "surgical box"
box_width = int(frame_width * 0.6) 
box_height = int(frame_height * 0.95)  
x1 = (frame_width - box_width) // 2
y1 = (frame_height - box_height) // 2
x2 = x1 + box_width
y2 = y1 + box_height
surgical_box = (x1, y1, x2, y2)


object_count = {}
object_tracker = {}

def is_within_box(xyxy, box):
    """Check if the detected object is within the defined box region."""
    x1, y1, x2, y2 = xyxy
    box_x1, box_y1, box_x2, box_y2 = box
    return (x1 > box_x1 and y1 > box_y1 and x2 < box_x2 and y2 < box_y2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference
    results = model(frame)

    # Draw the surgical box
    cv2.rectangle(frame, (surgical_box[0], surgical_box[1]), (surgical_box[2], surgical_box[3]), (0, 255, 0), 2)

    # Track objects in the current frame
    current_objects = {}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls = box.cls.item()
            label = model.names[int(cls)]
            xyxy = (x1, y1, x2, y2)

            # Check if the object is within the surgical box
            if is_within_box(xyxy, surgical_box):
                if label not in current_objects:
                    current_objects[label] = 0
                current_objects[label] += 1


                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    for obj, count in current_objects.items():
        if obj in object_tracker:

            object_tracker[obj] = max(object_tracker[obj], count)
        else:

            object_tracker[obj] = count


    object_count = {obj: count for obj, count in object_tracker.items() if count > 0}


    for obj in list(object_tracker.keys()):
        if obj not in current_objects:
            object_tracker[obj] -= 1  
            if object_tracker[obj] <= 0:
                del object_tracker[obj]

    
    y_offset = 20
    for obj, count in object_count.items():
        cv2.putText(frame, f'{obj}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        y_offset += 20

    # Show the frame
    cv2.imshow('Utensil Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()