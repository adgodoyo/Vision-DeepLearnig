# src/sort.py

import numpy as np
from filterpy.kalman import KalmanFilter

class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_id = 0

    def update(self, dets=np.empty((0, 5))):
        results = []
        for i, d in enumerate(dets):
            x1, y1, x2, y2, score = d
            results.append([x1, y1, x2, y2, self.track_id])
            self.track_id += 1
        return np.array(results)
