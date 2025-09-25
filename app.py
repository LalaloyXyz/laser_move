from typing import Optional

import cv2

# Local imports kept absolute so the module can run as a script
from detector import PersonDetector
from tracker_wrapper import MultiObjectTracker
from ui import PositionUI

class TrackingApp:
    def __init__(self,
                 camera_index: int = 0,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.4) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.detector = PersonDetector(model_path=model_path, confidence_threshold=confidence_threshold)
        self.tracker = MultiObjectTracker()
        self.ui = PositionUI()
        self.selected_id: Optional[int] = None

        cv2.namedWindow("Tracker")
        cv2.setMouseCallback("Tracker", self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN and hasattr(self, "_latest_tracks"):
            best_id, best_dist = None, 1e9
            for t in self._latest_tracks:
                if not t.is_confirmed() or t.time_since_update > 0:
                    continue
                l, t_, r, b = t.to_ltrb()
                cx, cy = (l + r) // 2, (t_ + b) // 2
                dist = (cx - x) ** 2 + (cy - y) ** 2
                if dist < best_dist:
                    best_dist, best_id = dist, t.track_id
            if best_id is not None:
                # Toggle selection if clicking the same id again
                if self.selected_id is not None and best_id == self.selected_id:
                    self.selected_id = None
                    self.ui.set_selected(None)
                else:
                    self.selected_id = best_id
                    self.ui.set_selected(best_id)

    def run(self) -> None:
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
                self._latest_tracks = tracks

                for t in tracks:
                    l, t_, r, b = map(int, t.to_ltrb())
                    tid = t.track_id
                    color = (0, 255, 0) if (self.selected_id is not None and tid == self.selected_id) else (128, 128, 128)
                    cv2.rectangle(frame, (l, t_), (r, b), color, 2)
                    cv2.putText(frame, f"ID {tid}", (l, t_ - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("Tracker", frame)
                self.ui.update_positions(tracks, frame.shape)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key in (ord('m'), ord('M')):
                    self.ui.toggle_mode()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.ui.destroy()


