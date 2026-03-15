"""Shared video I/O and metadata helpers (used by GUI and batch tools)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


@contextmanager
def _silence_stderr():
    """Temporarily silence stderr (used to hide OpenCV warnings when probing videos)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull)


def load_frames_bgr_range(
    avi_path: str,
    start_frame: int,
    end_frame: int,
) -> list[np.ndarray]:
    """Load frames [start_frame, end_frame] inclusive (0-based) as BGR uint8."""
    with _silence_stderr():
        cap = cv2.VideoCapture(str(avi_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(0, min(start_frame, total - 1))
        end = min(end_frame, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: list[np.ndarray] = []
        for _ in range(end - start + 1):
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()
    return frames


def get_video_frame_count(avi_path: str) -> int:
    """Total frame count; 0 if unreadable."""
    with _silence_stderr():
        cap = cv2.VideoCapture(str(avi_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return max(0, n)


def get_video_fps(avi_path: str) -> float:
    """Frame rate in Hz; 0 if unreadable."""
    with _silence_stderr():
        cap = cv2.VideoCapture(str(avi_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    return max(0.0, fps)


def load_metadata_for_video(video_path: str) -> dict:
    """
    Load metadata.json from the same directory as the video file.
    Returns dict with keys such as fps, frame_rate, etc. Empty dict if not found or invalid.
    """
    path = Path(video_path).resolve()
    meta_file = path.parent / "metadata.json"
    if not meta_file.is_file():
        return {}
    try:
        with open(meta_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

