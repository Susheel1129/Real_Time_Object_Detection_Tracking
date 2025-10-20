from tracker import CentroidTracker
from video_stream import VideoStream
from vis import draw_boxes
from ultralytics import YOLO
import cv2
import time

# Initialize YOLO
model = YOLO("models/yolov8n.pt")

# Tracker
tracker = CentroidTracker(max_disappeared=50)

# Webcam
vs = VideoStream(source=0).start()
time.sleep(1.0)  # allow camera warm-up

prev_time = 0

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # YOLO detection
    results = model(frame)[0]

    detections = []
    centroids = []

    for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.conf.cpu().numpy(),
                              results.boxes.cls.cpu().numpy()):
        if conf < 0.3:  # filter low-confidence detections
            continue
        x1, y1, x2, y2 = map(int, box)
        detections.append([x1, y1, x2, y2, conf, cls])
        centroids.append([x1, y1, x2, y2])

    # Update tracker
    tracks = tracker.update(centroids)

    # Draw
    draw_boxes(frame, tracks, detections=detections, class_names=model.names, fps=fps)

    cv2.imshow("YOLO + Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
