from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import torch
from ultralytics import YOLO

Detection = Tuple[List[int], float, Optional[None]]

class PersonDetector:
    """Thin wrapper around YOLO person detection.

    Produces detections in DeepSORT's expected format: ([x, y, w, h], conf, class)
    where class is unused here (kept as None).
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.4) -> None:
        self.confidence_threshold = confidence_threshold
        if model_path is None:
            # Resolve bundled model relative to this file
            model_path = str(Path(__file__).resolve().parent / "Model" / "yolo12n.pt")
        self.model = YOLO(model_path)
        # Force CPU to match user's environment preference
        self.device = "cpu"
        self.use_half = False

    def detect(self, frame) -> List[Detection]:
        """Return person detections in [x, y, w, h] format with confidences."""
        # Ensure frame is a proper image
        if frame is None:
            return []

        # Use smaller inference settings on CPU for speed and filter to person class
        # Power-saving optimizations: smaller image, lower confidence, single class
        results = self.model(frame, device=self.device, half=self.use_half, verbose=False, 
                           imgsz=480, classes=[0], conf=0.3, iou=0.5)[0]
        detections: List[Detection] = []

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            # COCO person class id == 0
            if int(cls) == 0 and float(conf) >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), None))

        return detections


