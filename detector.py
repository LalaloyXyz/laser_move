from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

Detection = Tuple[List[int], float, Optional[None]]

# Pose keypoint connections (skeleton)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]


class PersonDetector:
    """Pose-based person detection using YOLO pose model.

    Produces detections in DeepSORT's expected format: ([x, y, w, h], conf, class)
    where class is unused here (kept as None).
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.4) -> None:
        self.confidence_threshold = confidence_threshold

        if model_path is None:
            # Default model path
            model_path = str(Path(__file__).resolve().parent / "Model" / "yolo11n-pose.pt")

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Pose model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading pose model from {model_path}: {e}")
            raise RuntimeError(f"Could not load pose model from {model_path}")

        self.last_keypoints = None

    def detect(self, frame) -> List[Detection]:
        """Improved detection with fused keypoint + YOLO box info."""
        if frame is None:
            return []

        detections: List[Detection] = []
        results = self.model(
            frame,
            device="cpu",
            half=False,
            verbose=False,
            imgsz=352,
            classes=[0],
            conf=self.confidence_threshold,
            iou=0.1
        )[0]

        if results.keypoints is not None:
            for i, keypoints in enumerate(results.keypoints.xy):
                if len(keypoints) == 0:
                    continue

                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()

                keypoints_list = []
                for kp in keypoints:
                    if kp[0] > 1 and kp[1] > 1:
                        keypoints_list.append([float(kp[0]), float(kp[1]), 1.0])
                    else:
                        keypoints_list.append([0.0, 0.0, 0.0])
                self.last_keypoints = keypoints_list

                valid_points = keypoints[keypoints[:, 0] > 1]
                if len(valid_points) > 0:
                    x1 = max(0, int(np.min(valid_points[:, 0]) - 15))
                    y1 = max(0, int(np.min(valid_points[:, 1]) - 15))
                    x2 = min(frame.shape[1], int(np.max(valid_points[:, 0]) + 15))
                    y2 = min(frame.shape[0], int(np.max(valid_points[:, 1]) + 15))

                    kp_conf = np.mean([c for (_, _, c) in keypoints_list if c > 0])
                    box_conf = float(results.boxes.conf[i]) if i < len(results.boxes.conf) else kp_conf
                    conf = max(box_conf, kp_conf)

                    if conf >= self.confidence_threshold and (x2 - x1) > 20 and (y2 - y1) > 20:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

        return detections

    def get_head_position(self, frame, detection_box: List[int]) -> Optional[Tuple[int, int]]:
        """Get head position using pose keypoints."""
        if self.last_keypoints is None:
            return None

        keypoints = self.last_keypoints

        # Nose first
        if len(keypoints) > 0 and keypoints[0][2] > 0.3:
            return (int(keypoints[0][0]), int(keypoints[0][1]))

        # Eyes
        if len(keypoints) > 2 and keypoints[1][2] > 0.3:
            return (int(keypoints[1][0]), int(keypoints[1][1]))
        if len(keypoints) > 2 and keypoints[2][2] > 0.3:
            return (int(keypoints[2][0]), int(keypoints[2][1]))

        # Ears
        if len(keypoints) > 4 and keypoints[3][2] > 0.3:
            return (int(keypoints[3][0]), int(keypoints[3][1]))
        if len(keypoints) > 4 and keypoints[4][2] > 0.3:
            return (int(keypoints[4][0]), int(keypoints[4][1]))

        return None

    def draw_keypoints(self, frame, detection_box=None, confidence=0.0):
        """Draw bounding box, skeleton (white), keypoints (red, nose yellow)."""
        global last_nose_position, nose_debug

        if self.last_keypoints is None:
            return frame

        keypoints = self.last_keypoints
        if sum(1 for (_, _, c) in keypoints if c > 0.3) < 3:
            return frame

        # Bounding box
        if detection_box:
            x, y, w, h = detection_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        # Skeleton
        for kp1, kp2 in POSE_CONNECTIONS:
            if kp1 < len(keypoints) and kp2 < len(keypoints):
                x1, y1, c1 = keypoints[kp1]
                x2, y2, c2 = keypoints[kp2]
                if c1 > 0.3 and c2 > 0.3:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # Keypoints
        for i, (kx, ky, kc) in enumerate(keypoints):
            if kc > 0.3:
                color = (0, 0, 255)
                cv2.circle(frame, (int(kx), int(ky)), 7, (0, 0, 0), -1)
                cv2.circle(frame, (int(kx), int(ky)), 5, color, -1)

        return frame
