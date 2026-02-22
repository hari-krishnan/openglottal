"""Vocal fold tracking models: VocalFoldTracker and YOLOGuidedVFT."""

from __future__ import annotations

import cv2
import numpy as np

from ..utils import _resize_to


class VocalFoldTracker:
    """
    Motion-based glottal area segmentation.

    Based on: Unnikrishnan (2016). Operates on grayscale crops.

    All frames passed to :meth:`initialize` and :meth:`process_frame` are
    normalised to the size of the first init frame, so ``cv2.absdiff`` never
    sees a shape mismatch.

    Parameters
    ----------
    alpha:
        EMA weight for the motion map (higher = more temporal smoothing).
    beta:
        EMA weight for the intensity threshold.
    roi_threshold_ratio:
        Fraction of peak motion-map value used as the ROI threshold.
    gaussian_ksize:
        Kernel size for Gaussian blur (must be odd).
    glottal_percentile:
        Percentile of pixels inside the ROI used to seed the threshold.
    max_glottal_components:
        Maximum number of connected components kept in the output mask.
    """

    def __init__(
        self,
        alpha: float = 0.98,
        beta: float = 0.7,
        roi_threshold_ratio: float = 0.07,
        gaussian_ksize: int = 13,
        glottal_percentile: int = 5,
        max_glottal_components: int = 2,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.roi_ratio = roi_threshold_ratio
        self.gk = (gaussian_ksize, gaussian_ksize)
        self.pct = glottal_percentile
        self.n_comp = max_glottal_components
        self.prev = self.lmap = self.thresh = self.rthr = None
        self._w = self._h = None  # locked frame size

    # ── internal helpers ──────────────────────────────────────────────────────

    def _blob(self, m: np.ndarray) -> np.ndarray:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return np.zeros_like(m)
        out = np.zeros_like(m)
        cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        return out

    def _nblobs(self, m: np.ndarray, n: int) -> np.ndarray:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return np.zeros_like(m)
        out = np.zeros_like(m)
        cv2.drawContours(
            out,
            sorted(cs, key=cv2.contourArea, reverse=True)[:n],
            -1,
            255,
            cv2.FILLED,
        )
        return out

    # ── public API ────────────────────────────────────────────────────────────

    def initialize(self, frames: list[np.ndarray]) -> None:
        """Seed the motion map and intensity threshold from a list of grayscale frames."""
        self._h, self._w = frames[0].shape[:2]
        ff = [_resize_to(f, self._w, self._h).astype(np.float32) for f in frames]

        avg = sum(cv2.absdiff(ff[i], ff[i - 1]) for i in range(1, len(ff))) / (len(ff) - 1)
        self.lmap = cv2.GaussianBlur(avg, self.gk, 0)
        peak = self.lmap.max()
        self.rthr = peak * self.roi_ratio if peak > 0 else 1.0
        _, rm = cv2.threshold(self.lmap, self.rthr, 255, cv2.THRESH_BINARY)
        rm = self._blob(rm.astype(np.uint8))
        px = frames[0][rm == 255]
        self.thresh = float(np.percentile(px, self.pct)) if px.size > 0 else 127.0
        self.prev = ff[-1]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one grayscale frame and return a binary uint8 mask (255 = glottis).
        """
        frame = _resize_to(frame, self._w, self._h)
        ff = frame.astype(np.float32)
        d = cv2.GaussianBlur(cv2.absdiff(ff, self.prev), self.gk, 0)
        self.lmap = self.alpha * d + (1 - self.alpha) * self.lmap
        peak = self.lmap.max()
        self.rthr = peak * self.roi_ratio if peak > 0 else 1.0
        _, rr = cv2.threshold(self.lmap, self.rthr, 255, cv2.THRESH_BINARY)
        roi = self._blob(rr.astype(np.uint8))
        px = frame[roi == 255]
        cur = float(np.percentile(px, self.pct)) if px.size > 10 else self.thresh
        self.thresh = self.beta * self.thresh + (1 - self.beta) * cur
        raw = np.zeros_like(frame, dtype=np.uint8)
        raw[(frame < self.thresh) & (roi == 255)] = 255
        self.prev = ff
        return self._nblobs(raw, self.n_comp)


class YOLOGuidedVFT:
    """
    VocalFoldTracker with a YOLO-provided ROI.

    The YOLO bounding box replaces VFT's internal motion-map ROI detection
    entirely. Each frame the YOLO detector returns a ``(x1, y1, x2, y2)``
    box; this is used as a binary mask restricting both the intensity
    threshold estimation and the output mask.

    What is kept from VFT:
      - EMA motion map for temporal smoothing of frame differences
      - Beta-smoothed intensity threshold adapted to pixels inside the
        current YOLO bbox
      - N-largest-blob filtering on the output mask

    What is removed compared to VFT:
      - ``rthr`` / ``roi_ratio`` → motion-map ROI blob detection
      - ``_blob()`` for ROI extraction
      - Crop resizing / size-locking workarounds

    Result: stable ROI from frame 0, no warmup, no crop-size drift.
    """

    def __init__(
        self,
        alpha: float = 0.98,
        beta: float = 0.7,
        glottal_percentile: int = 5,
        gaussian_ksize: int = 13,
        max_glottal_components: int = 2,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.pct = glottal_percentile
        self.gk = (gaussian_ksize, gaussian_ksize)
        self.n_comp = max_glottal_components
        self.prev = None
        self.lmap = None
        self.thresh = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bbox_mask(self, shape: tuple, bbox: tuple | None) -> np.ndarray:
        """Return a uint8 mask that is 255 inside ``bbox``, 0 outside."""
        m = np.zeros(shape[:2], np.uint8)
        if bbox is not None:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            m[y1:y2, x1:x2] = 255
        return m

    def _nblobs(self, m: np.ndarray, n: int) -> np.ndarray:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return np.zeros_like(m)
        out = np.zeros_like(m)
        cv2.drawContours(
            out,
            sorted(cs, key=cv2.contourArea, reverse=True)[:n],
            -1,
            255,
            cv2.FILLED,
        )
        return out

    # ── API ───────────────────────────────────────────────────────────────────

    def initialize(self, frames: list[np.ndarray], bbox: tuple | None = None) -> None:
        """
        Seed motion map and intensity threshold.

        Parameters
        ----------
        frames:
            Grayscale uint8 arrays (full frame, any size).
        bbox:
            ``(x1, y1, x2, y2)`` from first YOLO detection; if ``None``
            uses the full frame.
        """
        ff = [f.astype(np.float32) for f in frames]
        diffs = [cv2.absdiff(ff[i], ff[i - 1]) for i in range(1, len(ff))]
        avg = sum(diffs) / len(diffs)
        self.lmap = cv2.GaussianBlur(avg, self.gk, 0)

        roi = self._bbox_mask(frames[0].shape, bbox)
        px = frames[-1][roi == 255] if roi.any() else frames[-1].ravel()
        self.thresh = float(np.percentile(px, self.pct)) if px.size > 0 else 127.0
        self.prev = ff[-1]

    def process_frame(self, frame: np.ndarray, bbox: tuple | None) -> np.ndarray:
        """
        Parameters
        ----------
        frame:
            Grayscale uint8 full-resolution frame (e.g. 256×256).
        bbox:
            ``(x1, y1, x2, y2)`` from :class:`TemporalDetector` for this
            frame, or ``None`` (produces an empty mask).

        Returns
        -------
        Binary uint8 mask (255 = glottis), same shape as ``frame``.
        """
        ff = frame.astype(np.float32)
        d = cv2.GaussianBlur(cv2.absdiff(ff, self.prev), self.gk, 0)
        self.lmap = self.alpha * d + (1 - self.alpha) * self.lmap

        roi = self._bbox_mask(frame.shape, bbox)
        px = frame[roi == 255]
        cur = float(np.percentile(px, self.pct)) if px.size > 10 else self.thresh
        self.thresh = self.beta * self.thresh + (1 - self.beta) * cur

        raw = np.zeros_like(frame, dtype=np.uint8)
        raw[(frame < self.thresh) & (roi == 255)] = 255

        self.prev = ff
        return self._nblobs(raw, self.n_comp)
