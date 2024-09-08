import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using a pre-trained YOLOv8 model

# Initialize DeepSORT for object tracking
tracker = DeepSort(max_age=30, nn_budget=70)

# Function to classify objects as "child" or "therapist" (Can be based on heuristics or pose estimations)
def classify_person(bbox, person_id):
    # Placeholder classification logic. Ideally, this would use behavioral cues.
    if person_id % 2 == 0:  # Let's assume alternating IDs
        return "child"
    else:
        return "therapist"

# Function to process video and overlay predictions
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 inference for person detection
        results = model(frame)
        detections = []
        
        # Collect detections (YOLO returns results in XYXY format)
        for result in results:
            for det in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                    detections.append([x1, y1, x2 - x1, y2 - y1, conf])

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and labels
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom bounding box coordinates
            x1, y1, x2, y2 = map(int, ltrb)
            label = classify_person(ltrb, track_id)
            color = (0, 255, 0) if label == "child" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label and ID
            text = f"{label} {track_id}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Write the frame with detections
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Path to your input video and output file
video_path = input("Enter your video path:") # Enter your video path here
output_path = "output_video.mp4"

# Run the processing
process_video(video_path, output_path)

# Output Video with predictions is saved as "output_video.mp4"
