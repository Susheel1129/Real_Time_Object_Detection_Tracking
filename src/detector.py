# detector.py
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path=r"C:\Users\ksush\Downloads\Summer_intern\Real_time_object_detection_tracking\runs\yolov8_finetune_v2\weights\best.pt",
                 conf_threshold=0.5):
        """
        Initialize YOLOv8 detector.
        model_path: path to YOLO weights (.pt)
        conf_threshold: confidence threshold for detection filtering
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_objects(self, frame):
        """
        Run object detection on a single frame.
        Returns: list of detections [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append([
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                    float(box.conf[0]),
                    int(box.cls[0]),
                ])
        return detections
