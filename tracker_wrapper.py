from typing import Any, Dict, List

import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiObjectTracker:
    """Wrapper for DeepSort tracker to simplify updates and output format."""

    def __init__(self,
                 max_age: int = 30,
                 n_init: int = 1,
                 max_iou_distance: float = 0.6) -> None:
        # If deep_sort_realtime was installed with a Torch embedder, this will pick GPU automatically
        # by reading torch.cuda.is_available(). Expose device explicitly to be safe.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tracker = DeepSort(max_age=max_age,
                                 n_init=n_init,
                                 max_iou_distance=max_iou_distance,
                                 embedder_gpu=(device == 'cuda'))

    def update(self, detections: List, frame: np.ndarray):
        """Update tracker and return confirmed tracks only."""
        tracks = self._tracker.update_tracks(detections, frame=frame)
        confirmed_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]
        return confirmed_tracks

    @staticmethod
    def to_center_normalized(track, frame_shape) -> Dict[str, float]:
        """Return normalized center coords for a track in range [-1, 1]."""
        h, w = frame_shape[:2]
        l, t_, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t_ + b) // 2
        nx = (cx / w) * 2 - 1
        ny = -((cy / h) * 2 - 1)
        return {"x": nx, "y": ny}


