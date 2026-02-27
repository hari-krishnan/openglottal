"""Shared utilities: I/O, frame helpers, segmentation metrics."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch


# ── Weight path resolution ───────────────────────────────────────────────────

def resolve_weights_path(path: str | Path) -> Path:
    """Return path if it exists; else try weights/<basename> for legacy / in-progress runs."""
    p = Path(path)
    if p.exists():
        return p
    legacy = Path("weights") / p.name
    if legacy.exists():
        return legacy
    return p


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


# ── Letterbox (aspect-ratio preserving crop resize) ───────────────────────────

def letterbox(
    img: np.ndarray,
    size: int = 256,
    value: int = 0,
) -> np.ndarray:
    """
    Scale img so its longest side = ``size``, then pad symmetrically to
    produce a square ``size``×``size`` array. Preserves aspect ratio.

    Works for 2-D (grayscale/mask) and 3-D (BGR) arrays.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_h = size - new_h
    pad_w = size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    if img.ndim == 3:
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(value, value, value),
        )
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=value,
    )


def letterbox_with_info(
    img: np.ndarray,
    size: int = 256,
    value: int = 0,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Same as ``letterbox`` but also return geometry for unletterbox/resize-back.

    Returns
    -------
    letterboxed : np.ndarray
        Padded square image.
    pad_top, pad_left : int
        Top/left padding (content region starts here).
    content_h, content_w : int
        Height/width of the scaled content inside the square.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_h = size - new_h
    pad_w = size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    if img.ndim == 3:
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(value, value, value),
        )
    else:
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=value,
        )
    return out, top, left, new_h, new_w


def letterbox_apply_geometry(
    img: np.ndarray,
    size: int,
    pad_top: int,
    pad_left: int,
    content_h: int,
    content_w: int,
    value: int = 0,
    interp: int | None = None,
) -> np.ndarray:
    """
    Resize and pad ``img`` to (size, size) using the same geometry as a
    previous ``letterbox_with_info`` call. Use for masks (interp=INTER_NEAREST).
    """
    if interp is None:
        interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (content_w, content_h), interpolation=interp)
    pad_bottom = size - pad_top - content_h
    pad_right = size - pad_left - content_w
    if img.ndim == 3:
        return cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(value, value, value),
        )
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=value,
    )


def unletterbox(
    letterboxed: np.ndarray,
    pad_top: int,
    pad_left: int,
    content_h: int,
    content_w: int,
    target_h: int,
    target_w: int,
    interp: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """
    Crop the content region from a letterboxed image and resize to target size.
    Use to project a model output mask back to original crop dimensions.
    """
    crop = letterboxed[
        pad_top : pad_top + content_h,
        pad_left : pad_left + content_w,
    ]
    if (content_h, content_w) == (target_h, target_w):
        return crop
    return cv2.resize(crop, (target_w, target_h), interpolation=interp)


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
