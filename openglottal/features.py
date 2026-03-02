"""Kinematic feature extraction across all three pipelines."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .models.tracker import VocalFoldTracker, YOLOGuidedVFT
from .utils import load_frames_bgr, unet_segment_frame

# ── Default tracker parameters ────────────────────────────────────────────────

VFT_PARAMS = dict(
    alpha=0.98,
    beta=0.7,
    roi_threshold_ratio=0.07,
    gaussian_ksize=13,
    glottal_percentile=5,
    max_glottal_components=2,
)

YGVFT_PARAMS = dict(
    alpha=0.98,
    beta=0.7,
    glottal_percentile=30,
    gaussian_ksize=13,
    max_glottal_components=2,
)

VFT_INIT = 2  # seed frames for Pipeline 1
YGVFT_INIT = 2  # seed frames for Pipeline 2


# ── Kinematic feature computation ─────────────────────────────────────────────


def _kinematic_features(area_wave: list[float]) -> dict | None:
    """
    Extract scalar kinematic features from a glottal area waveform.

    Returns ``None`` if the waveform is silent (all zeros).
    """
    area = np.array(area_wave)
    if area.max() == 0:
        return None
    mean_a = area.mean()
    std_a = area.std()
    oq = float(np.mean(area > mean_a * 0.1))
    fft = np.abs(np.fft.rfft(area - mean_a))
    freqs = np.fft.rfftfreq(len(area))
    peak_idx = int(np.argmax(fft[1:]) + 1)
    # First FFT bin (DC excluded) → no reliable f0
    f0: float | None = None if peak_idx == 1 else float(freqs[peak_idx])
    ac = np.correlate(area - mean_a, area - mean_a, mode="full")
    ac = ac[len(ac) // 2 :]
    ac /= ac[0] + 1e-8
    periodicity = float(ac[1 : min(50, len(ac))].max())
    return {
        "area_mean": mean_a,
        "area_std": std_a,
        "area_range": area.max() - area.min(),
        "open_quotient": oq,
        "f0": f0,
        "periodicity": periodicity,
        "cv": std_a / (mean_a + 1e-8),
        "_area": area,
    }


# ── Pipeline 1: YOLO + crop + VFT ────────────────────────────────────────────


def extract_features_detector(
    avi_path: str,
    detector,
    vft_init: int = VFT_INIT,
) -> dict | None:
    """
    Pipeline 1: YOLO (per frame) → fixed-size crop → VFT inside crop → area waveform.

    All crops are normalised to the locked size before being passed to VFT so
    that ``cv2.absdiff`` never sees a shape mismatch regardless of detector
    version.

    Parameters
    ----------
    avi_path:
        Path to the input video file.
    detector:
        A :class:`~openglottal.models.TemporalDetector` instance.
    vft_init:
        Number of seed frames used to initialise the tracker.

    Returns
    -------
    Dict of kinematic features, or ``None`` if the video is too short / silent.
    """
    frames_bgr = load_frames_bgr(avi_path)
    if len(frames_bgr) < vft_init + 5:
        return None

    detector.reset()
    tracker = None
    init_buf: list[np.ndarray] = []
    area_wave: list[float] = []
    target_hw: tuple[int, int] | None = None  # (w, h) locked on first valid crop

    for frm_bgr in frames_bgr:
        box = detector.detect(frm_bgr)

        if box is None:
            area_wave.append(0.0)
            continue

        x1, y1, x2, y2 = box
        crop_bgr = frm_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            area_wave.append(0.0)
            continue

        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        if target_hw is None:
            target_hw = (crop_gray.shape[1], crop_gray.shape[0])  # (w, h)
        elif crop_gray.shape != (target_hw[1], target_hw[0]):
            crop_gray = cv2.resize(crop_gray, target_hw, interpolation=cv2.INTER_LINEAR)

        if tracker is None:
            init_buf.append(crop_gray)
            if len(init_buf) >= vft_init:
                tracker = VocalFoldTracker(**VFT_PARAMS)
                tracker.initialize(init_buf)
                area_wave.extend([0.0] * len(init_buf))
                init_buf = []
            continue

        mask = tracker.process_frame(crop_gray)
        area_wave.append(float(np.sum(mask > 0)))

    return _kinematic_features(area_wave)


# ── Pipeline 2: YOLO-guided VFT ──────────────────────────────────────────────


def extract_features_yolo_guided_vft(
    avi_path: str,
    detector,
    ygvft_init: int = YGVFT_INIT,
) -> dict | None:
    """
    Pipeline 2: YOLO (per frame) → bbox as ROI mask on full frame →
    YOLOGuidedVFT → area waveform → kinematic features.

    No cropping, no size-locking. Only ``ygvft_init`` seed frames needed.
    When YOLO has no detection, the tracker uses an empty ROI → area = 0.

    Parameters
    ----------
    avi_path:
        Path to the input video file.
    detector:
        A :class:`~openglottal.models.TemporalDetector` instance.
    ygvft_init:
        Number of seed frames used to initialise the tracker.
    """
    frames_bgr = load_frames_bgr(avi_path)
    if len(frames_bgr) < ygvft_init + 2:
        return None

    detector.reset()
    tracker = None
    init_buf: list[np.ndarray] = []
    first_box: tuple | None = None
    area_wave: list[float] = []

    for frm_bgr in frames_bgr:
        gray = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2GRAY)
        box = detector.detect(frm_bgr)

        if tracker is None:
            init_buf.append(gray)
            if first_box is None and box is not None:
                first_box = box
            if len(init_buf) >= ygvft_init:
                tracker = YOLOGuidedVFT(**YGVFT_PARAMS)
                tracker.initialize(init_buf, bbox=first_box)
                area_wave.extend([0.0] * len(init_buf))
                init_buf = []
            continue

        mask = tracker.process_frame(gray, box)
        area_wave.append(float(np.sum(mask > 0)))

    return _kinematic_features(area_wave)


# ── Pipeline 3: YOLO + U-Net ─────────────────────────────────────────────────


def extract_features_unet(
    avi_path: str,
    detector,
    model: torch.nn.Module,
    device: torch.device,
) -> dict | None:
    """
    Pipeline 3 (YOLO+UNet) or U-Net-only: U-Net on full frame → area waveform →
    kinematic features. If detector is not None, only pixels inside the YOLO
    bbox are counted (detection-gated); if detector is None (unet-only), the
    full-frame mask sum is used every frame.

    Parameters
    ----------
    avi_path:
        Path to the input video file.
    detector:
        A :class:`~openglottal.models.TemporalDetector` instance, or None for
        unet-only (no YOLO gate; full-frame mask sum per frame).
    model:
        Trained :class:`~openglottal.models.UNet` in eval mode.
    device:
        Torch device to run inference on.
    """
    frames_bgr = load_frames_bgr(avi_path)
    if not frames_bgr:
        return None

    if detector is not None:
        detector.reset()
    area_wave: list[float] = []

    for frm_bgr in frames_bgr:
        gray_full = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2GRAY)
        mask_full = unet_segment_frame(gray_full, model, device)
        if detector is None:
            area_wave.append(float(np.sum(mask_full > 0)))
        else:
            box = detector.detect(frm_bgr)
            if box is None:
                area_wave.append(0.0)
            else:
                x1, y1, x2, y2 = box
                area_wave.append(float(np.sum(mask_full[y1:y2, x1:x2] > 0)))

    return _kinematic_features(area_wave)
