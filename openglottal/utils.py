"""Shared utilities: I/O, frame helpers, segmentation metrics."""

from __future__ import annotations

import contextlib
import os

import cv2
import numpy as np
import torch


# ── Frame I/O ────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_stderr():
    """Suppress OpenCV's noisy stderr warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull)


def load_frames_bgr(avi_path: str) -> list[np.ndarray]:
    """Load all frames from a video file as BGR uint8 arrays."""
    with _silence_stderr():
        cap = cv2.VideoCapture(str(avi_path))
        frames = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()
    return frames


def _resize_to(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize ``frame`` to ``(w, h)`` only if the current size differs."""
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


# ── Segmentation metrics ─────────────────────────────────────────────────────

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / union) if union > 0 else 1.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Dice loss for U-Net training."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


# ── U-Net inference helpers ───────────────────────────────────────────────────

def unet_segment_frame(
    frame_gray: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run U-Net on a ``(H, W)`` uint8 grayscale frame.

    The frame is resized to 256×256 for inference and the output mask is
    resized back to the original resolution.

    Returns
    -------
    Binary uint8 mask (255 = glottis), same shape as ``frame_gray``.
    """
    inp = cv2.resize(frame_gray, (256, 256), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(inp.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    H, W = frame_gray.shape
    if (H, W) != (256, 256):
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8) * 255
