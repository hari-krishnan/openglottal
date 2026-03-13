"""qt_app utilities: overlay and geometry. Imports shared helpers from openglottal as needed."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from openglottal.metadata import (
    load_frames_bgr_range,
    get_video_frame_count,
    get_video_fps,
    load_metadata_for_video,
)


def displacement_kinematic_features(
    left_disp: list[float],
    right_disp: list[float],
    fps: float,
) -> dict | None:
    """
    Kinematic features from left/right displacement buffer.
    Right is inverted; for Open Quotient anything negative is treated as zero (closed).
    Uses combined signal: width = left + (-right), then clip to >= 0.
    Returns dict with open_quotient, f0_hz, periodicity, cv, etc., or None if insufficient data.
    """
    if not left_disp or not right_disp or len(left_disp) != len(right_disp):
        return None
    n = len(left_disp)
    if n < 10:
        return None
    left = np.array(left_disp, dtype=np.float64)
    right = np.array(right_disp, dtype=np.float64)
    # Invert right; combined opening = left - right (width-like)
    right_inv = -right
    combined = left + right_inv
    # For OQ: anything negative is zero (closed)
    opening = np.maximum(combined, 0.0)
    return kinematic_features_from_opening(opening, fps)


def kinematic_features_from_opening(
    opening: np.ndarray | list[float],
    fps: float,
) -> dict | None:
    """
    Kinematic features from a single opening signal (e.g. width or area) sampled per frame.
    Opening should already be >= 0 (closed = 0). Returns same dict as displacement_kinematic_features.
    """
    opening = np.asarray(opening, dtype=np.float64)
    n = opening.size
    if n < 10:
        return None
    if opening.max() <= 0:
        return None
    mean_a = float(opening.mean())
    std_a = float(opening.std())
    # Open quotient: threshold from range so "closed" = near minimum (not 10% of mean which gives OQ=1)
    o_min, o_max = float(opening.min()), float(opening.max())
    o_range = o_max - o_min
    if o_range < 1e-9:
        oq = 0.0
    else:
        # Open when above 10% of range above minimum
        threshold = o_min + 0.1 * o_range
        oq = float(np.mean(opening > threshold))
    # F0 from FFT: rfftfreq(n) gives frequency in cycles per sample (= cycles per frame)
    # Hz = cycles_per_frame * frames_per_second (fps from user or metadata.json)
    detrended = opening - opening.mean()
    fft = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(n)  # cycles per frame
    peak_idx = int(np.argmax(fft[1:]) + 1)
    f0_cycles_per_frame: float | None = None if peak_idx <= 1 else float(freqs[peak_idx])
    f0_hz = (f0_cycles_per_frame * fps) if f0_cycles_per_frame is not None else None
    # Periodicity from autocorrelation
    ac = np.correlate(detrended, detrended, mode="full")
    ac = ac[len(ac) // 2 :]
    ac /= ac[0] + 1e-8
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


def overlay_segment(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay binary mask on BGR frame (green where mask > 0)."""
    out = frame_bgr.copy()
    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    glottis = mask > 0
    out[glottis] = (
        out[glottis].astype(np.float32) * (1 - alpha)
        + np.array([0, 255, 0], dtype=np.float32) * alpha
    ).astype(np.uint8)
    return out


def _ellipse_axes_from_mask(mask: np.ndarray):
    """
    Fit an ellipse to the largest contour; return center, semi-major and semi-minor axis endpoints.
    Returns (cx, cy, major_pt1, major_pt2, minor_pt1, minor_pt2) or None.
    Each pt is (x, y) float; axes extend from center along major/minor.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10 or len(contour) < 5:
        return None
    try:
        (cx, cy), (width, height), angle_deg = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    angle_rad = math.radians(angle_deg)
    semi_w = width / 2.0
    semi_h = height / 2.0
    if width >= height:
        major_len, minor_len = semi_w, semi_h
        major_angle = angle_rad
        minor_angle = angle_rad + math.pi / 2
    else:
        major_len, minor_len = semi_h, semi_w
        major_angle = angle_rad + math.pi / 2
        minor_angle = angle_rad
    cos_maj = math.cos(major_angle)
    sin_maj = math.sin(major_angle)
    cos_min = math.cos(minor_angle)
    sin_min = math.sin(minor_angle)
    major_pt1 = (cx + major_len * cos_maj, cy + major_len * sin_maj)
    major_pt2 = (cx - major_len * cos_maj, cy - major_len * sin_maj)
    # Use midpoint of major axis as center so minor axis passes through it
    cx = (major_pt1[0] + major_pt2[0]) / 2.0
    cy = (major_pt1[1] + major_pt2[1]) / 2.0
    minor_pt1 = (cx + minor_len * cos_min, cy + minor_len * sin_min)
    minor_pt2 = (cx - minor_len * cos_min, cy - minor_len * sin_min)
    return (cx, cy, major_pt1, major_pt2, minor_pt1, minor_pt2)


def _segment_line_intersection(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> tuple[float, float] | None:
    """
    Intersection of segment A->B with infinite line through C with direction D.
    Returns (x, y) if 0 <= s <= 1 for param s, else None.
    """
    vx = bx - ax
    vy = by - ay
    wx = cx - ax
    wy = cy - ay
    denom = vx * dy - vy * dx
    if abs(denom) < 1e-10:
        return None
    s = (wx * dy - wy * dx) / denom
    if s < -1e-6 or s > 1.0 + 1e-6:
        return None
    px = ax + s * vx
    py = ay + s * vy
    return (px, py)


def left_right_displacement_from_contour(
    contour: np.ndarray,
    cx: float,
    cy: float,
    angle_rad: float,
) -> tuple[float, float, tuple[float, float] | None, tuple[float, float] | None] | None:
    """
    Analytical intersection of contour (closed polygon) with minor axis line.
    Minor axis: line through (cx, cy) with direction perpendicular to major.
    For each contour segment, compute intersection with the line; collect points.
    Left = point with highest x, Right = point with lowest x.
    Returns (left_disp, right_disp, left_pt_xy, right_pt_xy) or None if < 2 intersections.
    """
    if contour is None or len(contour) < 3:
        return None
    pts = contour.reshape(-1, 2)
    cx_f, cy_f = float(cx), float(cy)
    cos_min = math.cos(angle_rad + math.pi / 2)
    sin_min = math.sin(angle_rad + math.pi / 2)
    intersections: list[tuple[float, float]] = []
    n = len(pts)
    for i in range(n):
        ax, ay = float(pts[i][0]), float(pts[i][1])
        j = (i + 1) % n
        bx, by = float(pts[j][0]), float(pts[j][1])
        hit = _segment_line_intersection(ax, ay, bx, by, cx_f, cy_f, cos_min, sin_min)
        if hit is not None:
            intersections.append(hit)
    if len(intersections) < 2:
        return None
    left_pt = max(intersections, key=lambda p: p[0])
    right_pt = min(intersections, key=lambda p: p[0])
    if abs(left_pt[0] - right_pt[0]) < 1e-3:
        return None
    left_disp = float(left_pt[0] - cx_f)
    right_disp = float(right_pt[0] - cx_f)
    return (left_disp, right_disp, left_pt, right_pt)


def left_right_displacement_from_mask(
    mask: np.ndarray,
    cx: float,
    cy: float,
    angle_rad: float,
) -> tuple[float, float, tuple[float, float] | None, tuple[float, float] | None] | None:
    """
    Uses contour from mask: largest contour by area, then analytical segment-line
    intersection with minor axis. When no contour, returns (0.0, 0.0, None, None).
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0.0, 0.0, None, None)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return (0.0, 0.0, None, None)
    return left_right_displacement_from_contour(contour, cx, cy, angle_rad)


def _draw_x_marker(bgr: np.ndarray, pt: tuple[float, float], color: tuple[int, int, int], thickness: int = 1, size: int = 5) -> None:
    """Draw a small x (tilted cross) at pt; minimal for few-pixel frames."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.drawMarker(bgr, (x, y), color, cv2.MARKER_TILTED_CROSS, size, thickness)


def draw_displacement_points_on_bgr(
    bgr: np.ndarray,
    left_pt: tuple[float, float] | None,
    right_pt: tuple[float, float] | None,
    radius: int = 6,
    color_left: tuple[int, int, int] = (255, 100, 100),
    color_right: tuple[int, int, int] = (100, 100, 255),
    thickness: int = 1,
) -> None:
    """Draw small x markers at the left and right displacement points (minimal for few-pixel frames)."""
    if left_pt is not None:
        _draw_x_marker(bgr, left_pt, color_left, thickness, size=5)
    if right_pt is not None:
        _draw_x_marker(bgr, right_pt, color_right, thickness, size=5)


def _line_clip_to_frame(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Extend the line through (x1,y1) and (x2,y2) and clip to frame [0,w] x [0,h].
    Returns the two endpoints (pt1, pt2) on the frame boundary, or None if degenerate.
    """
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    ts = []
    if abs(dx) >= 1e-9:
        for xedge in (0, w):
            t = (xedge - x1) / dx
            y = y1 + t * dy
            if 0 <= y <= h:
                ts.append((t, (xedge, int(round(y)))))
    if abs(dy) >= 1e-9:
        for yedge in (0, h):
            t = (yedge - y1) / dy
            x = x1 + t * dx
            if 0 <= x <= w:
                ts.append((t, (int(round(x)), yedge)))
    if len(ts) < 2:
        return None
    ts.sort(key=lambda p: p[0])
    return (ts[0][1], ts[-1][1])


def segments_to_ellipse_params(
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
    min_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float, float, float] | None:
    """
    Convert major/minor axis segments to ellipse params (cx, cy, angle_rad, semi_maj, semi_min).
    Center = intersection of the two lines; angle from major; semi-lengths from distance of endpoints to center.
    """
    (x1, y1), (x2, y2) = maj_seg
    (u1, v1), (u2, v2) = min_seg
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = u2 - u1, v2 - v1
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return None
    t = ((u1 - x1) * dy2 - (v1 - y1) * dx2) / denom
    cx = x1 + t * dx1
    cy = y1 + t * dy1
    angle = math.atan2(dy1, dx1)
    dmaj1 = math.hypot(x1 - cx, y1 - cy)
    dmaj2 = math.hypot(x2 - cx, y2 - cy)
    dmin1 = math.hypot(u1 - cx, v1 - cy)
    dmin2 = math.hypot(u2 - cx, v2 - cy)
    a = max(dmaj1, dmaj2)
    b = max(dmin1, dmin2)
    return (cx, cy, angle, a, b)


def ellipse_params_to_segments(
    cx: float,
    cy: float,
    angle_rad: float,
    semi_maj: float,
    semi_min: float,
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    """Convert ellipse params to major/minor axis segments clipped to frame."""
    W, H = frame_w - 1, frame_h - 1
    cos_maj = math.cos(angle_rad)
    sin_maj = math.sin(angle_rad)
    cos_min = math.cos(angle_rad + math.pi / 2)
    sin_min = math.sin(angle_rad + math.pi / 2)
    maj1 = (cx + semi_maj * cos_maj, cy + semi_maj * sin_maj)
    maj2 = (cx - semi_maj * cos_maj, cy - semi_maj * sin_maj)
    min1 = (cx + semi_min * cos_min, cy + semi_min * sin_min)
    min2 = (cx - semi_min * cos_min, cy - semi_min * sin_min)
    maj_end = _line_clip_to_frame(maj1[0], maj1[1], maj2[0], maj2[1], W, H)
    min_end = _line_clip_to_frame(min1[0], min1[1], min2[0], min2[1], W, H)
    return (maj_end or ((int(cx), int(cy)), (int(cx), int(cy))), min_end or ((int(cx), int(cy)), (int(cx), int(cy))))


def major_axis_segment_to_ac_pc_mid(
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    From major axis segment (medial line), return AC, PC, and mid.
    AC = anterior (min y), PC = posterior (max y). Mid = midpoint of the line segment AC–PC.
    """
    (x1, y1), (x2, y2) = maj_seg
    if y1 <= y2:
        ac, pc = (x1, y1), (x2, y2)
    else:
        ac, pc = (x2, y2), (x1, y1)
    mid = midpoint_of_segment(ac, pc)
    return (ac, pc, mid)


def medial_line_direction_and_half_length(
    ac: tuple[float, float], pc: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """
    Unit direction from AC to PC and half-length (for translation-only updates).
    Returns (d_unit, half_length) so that mid + half_length * d = PC and mid - half_length * d = AC.
    """
    dx = pc[0] - ac[0]
    dy = pc[1] - ac[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return ((0.0, 0.0), 0.0)
    d_unit = (dx / length, dy / length)
    half_length = length / 2.0
    return (d_unit, half_length)


def midpoint_of_segment(ac: tuple[float, float], pc: tuple[float, float]) -> tuple[float, float]:
    """
    Midpoint of the line segment joining AC and PC (medial line is not vertical always).
    Geometric center: halfway along the segment.
    """
    return ((ac[0] + pc[0]) / 2.0, (ac[1] + pc[1]) / 2.0)


def ac_pc_from_points_along_medial_line(
    points: np.ndarray,
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """
    AC and PC as the extreme points of the accumulated contour along the medial line.
    points: (N, 2) or (N, 1, 2) array of contour points.
    Projects onto medial direction; AC = extreme with min y (anterior), PC = max y (posterior).
    Returns (AC, PC) or None if degenerate.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    else:
        pts = pts.reshape(-1, 2)
    if len(pts) < 2:
        return None
    (x1, y1), (x2, y2) = maj_seg
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    d_unit = (dx / length, dy / length)
    mid_seg = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    proj = (pts[:, 0] - mid_seg[0]) * d_unit[0] + (pts[:, 1] - mid_seg[1]) * d_unit[1]
    i_min, i_max = int(np.argmin(proj)), int(np.argmax(proj))
    p_min = (float(pts[i_min, 0]), float(pts[i_min, 1]))
    p_max = (float(pts[i_max, 0]), float(pts[i_max, 1]))
    if p_min[1] <= p_max[1]:
        return (p_min, p_max)
    return (p_max, p_min)


def fit_ellipse_to_points(
    points: np.ndarray, frame_w: int, frame_h: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    """
    Fit one ellipse to all points (N,1,2); return major and minor axis segments extended to frame.
    Returns ((maj_pt1, maj_pt2), (min_pt1, min_pt2)) or None.
    """
    if points is None or len(points) < 5:
        return None
    try:
        (cx, cy), (width, height), angle_deg = cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return None
    angle_rad = math.radians(angle_deg)
    semi_w, semi_h = width / 2.0, height / 2.0
    if width >= height:
        major_len, minor_len = semi_w, semi_h
        major_angle = angle_rad
        minor_angle = angle_rad + math.pi / 2
    else:
        major_len, minor_len = semi_h, semi_w
        major_angle = angle_rad + math.pi / 2
        minor_angle = angle_rad
    W, H = frame_w - 1, frame_h - 1
    maj1 = (cx + major_len * math.cos(major_angle), cy + major_len * math.sin(major_angle))
    maj2 = (cx - major_len * math.cos(major_angle), cy - major_len * math.sin(major_angle))
    min1 = (cx + minor_len * math.cos(minor_angle), cy + minor_len * math.sin(minor_angle))
    min2 = (cx - minor_len * math.cos(minor_angle), cy - minor_len * math.sin(minor_angle))
    maj_end = _line_clip_to_frame(maj1[0], maj1[1], maj2[0], maj2[1], W, H)
    min_end = _line_clip_to_frame(min1[0], min1[1], min2[0], min2[1], W, H)
    return (maj_end, min_end)


def draw_axes_on_bgr(
    bgr: np.ndarray,
    global_axes: tuple[
        tuple[tuple[int, int], tuple[int, int]] | None,
        tuple[tuple[int, int], tuple[int, int]] | None,
    ],
    offset_xy: tuple[float, float] = (0.0, 0.0),
    color_major: tuple[int, int, int] = (0, 0, 255),
    color_minor: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> None:
    """Draw precomputed major and minor axis segments on bgr (in-place)."""
    maj_seg, min_seg = global_axes
    dx, dy = offset_xy
    def _shift(seg):
        (x1, y1), (x2, y2) = seg
        return ((int(round(x1 + dx)), int(round(y1 + dy))), (int(round(x2 + dx)), int(round(y2 + dy))))
    if maj_seg is not None:
        s = _shift(maj_seg)
        cv2.line(bgr, s[0], s[1], color_major, thickness)
    if min_seg is not None:
        s = _shift(min_seg)
        cv2.line(bgr, s[0], s[1], color_minor, thickness)


def ac_pc_from_segment_along_medial_line(
    mask: np.ndarray,
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """
    AC and PC as the lowest and highest points of the segment along the medial line.
    Projects contour points onto the medial direction; AC = extreme with min y (anterior), PC = max y (posterior).
    Returns (AC, PC) or None if no contour.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
    (x1, y1), (x2, y2) = maj_seg
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    d_unit = (dx / length, dy / length)
    mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    pts = contour.reshape(-1, 2).astype(np.float64)
    proj = (pts[:, 0] - mid[0]) * d_unit[0] + (pts[:, 1] - mid[1]) * d_unit[1]
    i_min, i_max = int(np.argmin(proj)), int(np.argmax(proj))
    p_min, p_max = (float(pts[i_min, 0]), float(pts[i_min, 1])), (float(pts[i_max, 0]), float(pts[i_max, 1]))
    if p_min[1] <= p_max[1]:
        ac, pc = p_min, p_max
    else:
        ac, pc = p_max, p_min
    return (ac, pc)


def draw_ac_pc_labels_on_bgr(
    bgr: np.ndarray,
    maj_seg: tuple[tuple[float, float], tuple[float, float]] | None,
    mask: np.ndarray | None = None,
    offset_xy: tuple[float, float] = (0.0, 0.0),
    color_ac: tuple[int, int, int] = (0, 255, 255),
    color_pc: tuple[int, int, int] = (255, 165, 0),
    thickness: int = 1,
    font_scale: float = 0.35,
    marker_size: int = 5,
) -> None:
    """
    Draw AC and PC as small x markers (minimal for few-pixel frames). If mask is provided,
    AC/PC are the extreme points of the segment along the medial line; else use major-axis endpoints.
    In-place.
    """
    if maj_seg is None:
        return
    if mask is not None:
        ac_pc = ac_pc_from_segment_along_medial_line(mask, maj_seg)
        if ac_pc is None:
            return
        ac, pc = ac_pc
    else:
        (x1, y1), (x2, y2) = maj_seg
        if y1 <= y2:
            ac, pc = (x1, y1), (x2, y2)
        else:
            ac, pc = (x2, y2), (x1, y1)
    dx, dy = offset_xy
    ac_pt = (ac[0] + dx, ac[1] + dy)
    pc_pt = (pc[0] + dx, pc[1] + dy)
    # Labels swapped so anatomy matches: AC marker at pc_pt, PC marker at ac_pt
    _draw_x_marker(bgr, pc_pt, color_ac, thickness, size=marker_size)
    _draw_x_marker(bgr, ac_pt, color_pc, thickness, size=marker_size)


def draw_midline_on_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """Draw ellipse major and minor axes (fitted to contour) extended to frame. In-place. No-op if no contour."""
    axes = _ellipse_axes_from_mask(mask)
    if axes is None:
        return
    cx, cy, maj1, maj2, min1, min2 = axes
    h, w = bgr.shape[:2]
    W, H = w - 1, h - 1
    maj_end = _line_clip_to_frame(maj1[0], maj1[1], maj2[0], maj2[1], W, H)
    min_end = _line_clip_to_frame(min1[0], min1[1], min2[0], min2[1], W, H)
    if maj_end is not None:
        cv2.line(bgr, maj_end[0], maj_end[1], color, thickness)
    if min_end is not None:
        cv2.line(bgr, min_end[0], min_end[1], (255, 0, 0), thickness)  # minor axis in blue
