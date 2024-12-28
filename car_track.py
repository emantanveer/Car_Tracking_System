import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")
# Input and output video paths
input_video_path = 'C:/Users/HP/Downloads/features/features/car_tracking/testa.mp4'
output_video_path = 'car_out.mp4'
# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = 640
frame_height = 640
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
# Define the circle's properties
circle_center = (frame_width // 2, frame_height // 2)  
circle_radius = 215  

car_status = {}
in_count = 0
out_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame=cv2.resize(frame,(640,640))
    if not ret:
        break
    
    results = model.track(source=frame, persist=True,conf=0.6, iou=0.5, show=True)  
    # Extract tracking results
    detections = results[0].boxes.xywh 
    object_ids = results[0].boxes.id  
    # Draw the circle
    cv2.circle(frame, circle_center, circle_radius, (0, 255, 0), 3)
    # Process each tracked object
    for i, detection in enumerate(detections):
        x, y, w, h = detection
        obj_id = object_ids[i].item()  # Extract object ID
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        # Calculate the centroid of the bounding box
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        # Check if the centroid is inside the circle
        distance_to_center = np.sqrt((centroid_x - circle_center[0])**2 + (centroid_y - circle_center[1])**2)
        status = "in" if distance_to_center <= circle_radius else "out"
        # Update car status and counts
        if obj_id not in car_status:
            car_status[obj_id] = {"status": status, "crossed_in": False, "crossed_out": False}
        else:
            previous_status = car_status[obj_id]["status"]
            if previous_status != status:
                # If the car crosses the circle, update the count
                if status == "in" and not car_status[obj_id]["crossed_in"]:
                    in_count += 1
                    car_status[obj_id]["crossed_in"] = True
                elif status == "out" and not car_status[obj_id]["crossed_out"]:
                    out_count += 1
                    car_status[obj_id]["crossed_out"] = True
        
        car_status[obj_id]["status"] = status
        # Draw bounding box and status
        color = (0, 255, 0) if status == "in" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {int(obj_id)} ({status})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 255), -1)
    # Display total counts on the frame
        cv2.putText(frame, f"Total In: {in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, f"Total Out: {out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    out.write(frame)

cap.release()
out.release()
print(f"Inference completed! Output saved to: {output_video_path}")