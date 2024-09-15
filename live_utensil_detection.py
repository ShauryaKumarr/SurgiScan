import cv2
from ultralytics import YOLO
import numpy as np

# Load your custom trained YOLOv5 model
model = YOLO('/opt/homebrew/runs/detect/train6/weights/best.pt')

# Initialize video capture (0 for webcam; or provide video file path)
cap = cv2.VideoCapture(0)

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to capture frame")

frame_height, frame_width = frame.shape[:2]

# Define the "surgical box"
box_width = int(frame_width * 0.6)  # 60% of the frame width
box_height = int(frame_height * 0.95)  # 95% of the frame height
x1 = (frame_width - box_width) // 2
y1 = (frame_height - box_height) // 2
x2 = x1 + box_width
y2 = y1 + box_height
surgical_box = (x1, y1, x2, y2)

# Dictionary to count objects inside the box
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

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update the object counts with a simple tracking mechanism
    for obj, count in current_objects.items():
        if obj in object_tracker:
            # If object is still in the box, update the count smoothly
            object_tracker[obj] = max(object_tracker[obj], count)
        else:
            # New object detected in the box
            object_tracker[obj] = count

    # Update the display count from the tracker
    object_count = {obj: count for obj, count in object_tracker.items() if count > 0}

    # Remove objects that have left the box
    for obj in list(object_tracker.keys()):
        if obj not in current_objects:
            object_tracker[obj] -= 1  # Gradually decrease the count
            if object_tracker[obj] <= 0:
                del object_tracker[obj]

    # Display counts on the frame
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