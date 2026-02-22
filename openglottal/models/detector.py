"""YOLOv8-based glottis detector with temporal consistency."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class TemporalDetector:
    """
    YOLOv8-based glottal bounding-box detector with temporal consistency.

    The box **centre** is drift-clamped (max ``max_shift_px`` per frame) to
    reject spurious jumps.  The box **size** is updated from every fresh
    detection (plus ``padding`` pixels on each side) so the box grows and
    shrinks with the glottis through its vibratory cycle.

    When YOLO misses a frame the previous centre and size are held for at most
    ``max_hold_frames`` consecutive misses; after that the detection is zeroed
    (returns None) until YOLO fires again.
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        max_shift_px: int = 30,
        padding: int = 8,
        max_hold_frames: int = 3,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.max_shift = max_shift_px
        self.padding = padding
        self.max_hold_frames = max_hold_frames
        self._prev_cx: float | None = None
        self._prev_cy: float | None = None
        self._cur_w: int | None = None   # current width  (updated each detection)
        self._cur_h: int | None = None   # current height (updated each detection)
        self._miss_count: int = 0       # consecutive frames with no fresh detection

    def reset(self) -> None:
        self._prev_cx = self._prev_cy = None
        self._cur_w = self._cur_h = None
        self._miss_count = 0

    @property
    def crop_size(self) -> tuple[int, int] | None:
        """(w, h) of the current crop, or None before first detection."""
        return (self._cur_w, self._cur_h) if self._cur_w is not None else None

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Returns ``(x1, y1, x2, y2)`` or ``None`` if no detection has been
        seen yet.  The box size reflects the current YOLO detection (+ padding).
        """
        H, W = frame_bgr.shape[:2]
        results = self.model(frame_bgr, conf=self.conf, verbose=False)
        boxes = results[0].boxes

        new_cx = new_cy = new_w = new_h = None
        if boxes is not None and len(boxes):
            idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
            new_cx = (x1 + x2) / 2
            new_cy = (y1 + y2) / 2
            p = self.padding
            new_w = int(x2 - x1) + 2 * p
            new_h = int(y2 - y1) + 2 * p

        # Temporal consistency: reject if centre jumped too far
        if new_cx is not None and self._prev_cx is not None:
            if np.hypot(new_cx - self._prev_cx, new_cy - self._prev_cy) > self.max_shift:
                new_cx = new_cy = new_w = new_h = None  # hold previous

        if new_cx is not None:
            self._prev_cx, self._prev_cy = new_cx, new_cy
            self._cur_w, self._cur_h = new_w, new_h
            self._miss_count = 0
        elif self._prev_cx is not None:
            self._miss_count += 1
            if self._miss_count > self.max_hold_frames:
                self._prev_cx = self._prev_cy = None
                self._cur_w = self._cur_h = None
                self._miss_count = 0
                return None  # zeroed after max_hold_frames misses

        if self._prev_cx is None:
            return None  # no detection yet

        # Clamp centre so box stays within the frame
        hw = self._cur_w // 2
        hh = self._cur_h // 2
        cx = int(np.clip(self._prev_cx, hw, W - hw))
        cy = int(np.clip(self._prev_cy, hh, H - hh))
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def crop(self, frame: np.ndarray, box: tuple | None) -> np.ndarray:
        if box is None:
            return frame
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]
