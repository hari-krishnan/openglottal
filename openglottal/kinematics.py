"""Kinematic features from L/R displacement or opening (area/width) signals. Used by displacement module and GUI."""

from __future__ import annotations

import numpy as np


def kinematic_features_from_opening(
    opening: np.ndarray | list[float],
    fps: float,
) -> dict | None:
    """
    Scalar kinematic features from an opening signal (≥0 per frame).
    Returns dict with open_quotient, f0_hz, periodicity, cv, mean_opening, std_opening, fps_used; or None.
    """
    opening = np.asarray(opening, dtype=np.float64)
    n = opening.size
    if n < 10 or opening.max() <= 0:
        return None
    mean_a = float(opening.mean())
    std_a = float(opening.std())
    o_min, o_max = float(opening.min()), float(opening.max())
    o_range = o_max - o_min
    if o_range < 1e-9:
        oq = 0.0
    else:
        threshold = o_min + 0.1 * o_range
        oq = float(np.mean(opening > threshold))
    detrended = opening - opening.mean()
    fft = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(n)
    peak_idx = int(np.argmax(fft[1:]) + 1)
    f0_cycles_per_frame: float | None = None if peak_idx <= 1 else float(freqs[peak_idx])
    f0_hz = (f0_cycles_per_frame * fps) if f0_cycles_per_frame is not None else None
    ac = np.correlate(detrended, detrended, mode="full")
    ac = ac[len(ac) // 2 :] / (ac[0] + 1e-8)
    periodicity = float(ac[1 : min(50, len(ac))].max())
    cv = std_a / (mean_a + 1e-8)
    return {
        "open_quotient": oq,
        "f0_hz": f0_hz,
        "fps_used": fps,
        "periodicity": periodicity,
        "cv": cv,
        "mean_opening": mean_a,
        "std_opening": std_a,
    }


def displacement_kinematic_features(
    left_disp: list[float],
    right_disp: list[float],
    fps: float,
) -> dict | None:
    """
    Kinematic features from L/R displacement buffers. Right is inverted; opening = max(L + (-R), 0).
    Returns same dict as kinematic_features_from_opening, or None.
    """
    if not left_disp or not right_disp or len(left_disp) != len(right_disp) or len(left_disp) < 10:
        return None
    left = np.array(left_disp, dtype=np.float64)
    right = np.array(right_disp, dtype=np.float64)
    opening = np.maximum(left + (-right), 0.0)
    return kinematic_features_from_opening(opening, fps)
