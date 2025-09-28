from typing import Any, Dict, List
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiObjectTracker:
    """Wrapper for DeepSort tracker with optional GPU support."""

    def __init__(self,
                 max_age: int = 30,
                 n_init: int = 1,
                 max_iou_distance: float = 0.5,
                 use_gpu: bool = True) -> None:
        """
        Initialize DeepSort tracker.

        :param max_age: Max frames to keep lost tracks.
        :param n_init: Minimum hits before confirming a track.
        :param max_iou_distance: IOU threshold for matching.
        :param use_gpu: Use GPU for embedding if available.
        """
        device_available = torch.cuda.is_available() and use_gpu
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder_gpu=device_available
        )
        print(f"[MultiObjectTracker] Using GPU: {device_available}")

    def update(self, detections: List, frame: np.ndarray) -> List:
        """
        Update tracker with current detections.

        :param detections: List of detections in [x, y, w, h, conf] format.
        :param frame: Current frame (BGR) for appearance embeddings.
        :return: List of confirmed tracks.
        """
        tracks = self._tracker.update_tracks(detections, frame=frame)
        # Only keep confirmed tracks updated this frame
        confirmed_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]
        return confirmed_tracks

    @staticmethod
    def to_center_normalized(track, frame_shape) -> Dict[str, float]:
        """
        Return normalized center coordinates in [-1, 1].

        :param track: A confirmed DeepSort track.
        :param frame_shape: Shape of the frame (H, W, C)
        :return: Dict with 'x' and 'y' in [-1, 1]
        """
        h, w = frame_shape[:2]
        l, t_, r, b = track.to_ltrb()
        cx = (l + r) / 2
        cy = (t_ + b) / 2
        nx = (cx / w) * 2 - 1
        ny = -((cy / h) * 2 - 1)
        return {"x": nx, "y": ny}
