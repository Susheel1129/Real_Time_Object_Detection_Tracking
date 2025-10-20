from src.tracker import CentroidTracker

tracker = CentroidTracker()

# Simulate object detection bounding boxes
frames = [
    [(30, 40, 80, 100)],
    [(35, 45, 85, 105)],
    [(100, 120, 150, 180)],
    [(105, 125, 155, 185)],
]

for frame_idx, rects in enumerate(frames):
    objects = tracker.update(rects)
    print(f"Frame {frame_idx + 1}: {objects}")
