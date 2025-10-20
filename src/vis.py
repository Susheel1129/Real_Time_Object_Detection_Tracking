import cv2

def draw_boxes(frame, tracks, detections=None, class_names=None, fps=None):
    """
    Draws tracked objects with stable bounding boxes, IDs, and class names.
    """
    if detections is not None and class_names is not None:
        for objectID, centroid in tracks.items():
            # Find closest detection
            closest_det = None
            min_dist = float('inf')
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cX = (x1 + x2) // 2
                cY = (y1 + y2) // 2
                dist = ((cX - centroid[0])**2 + (cY - centroid[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_det = det
            if closest_det:
                x1, y1, x2, y2, conf, cls = closest_det
                if conf < 0.3:  # confidence threshold
                    continue
                label = f"{class_names[int(cls)]} ID:{objectID} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Draw FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame
