"""Ellipse fitting, medial line (AC/PC), and L/R displacement from contours. Used by displacement module and GUI."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _segment_line_intersection(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> tuple[float, float] | None:
    """Intersection of segment A->B with infinite line through C with direction D. Returns (x, y) or None."""
    vx, vy = bx - ax, by - ay
    wx, wy = cx - ax, cy - ay
    denom = vx * dy - vy * dx
    if abs(denom) < 1e-10:
        return None
    s = (wx * dy - wy * dx) / denom
    if s < -1e-6 or s > 1.0 + 1e-6:
        return None
    return (ax + s * vx, ay + s * vy)


def left_right_displacement_from_contour(
    contour: np.ndarray,
    cx: float,
    cy: float,
    angle_rad: float,
) -> tuple[float, float, tuple[float, float] | None, tuple[float, float] | None] | None:
    """
    L/R displacement from contour: intersect contour with minor-axis line; left = max x, right = min x.
    Returns (left_disp, right_disp, left_pt_xy, right_pt_xy) or None.
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
    return (float(left_pt[0] - cx_f), float(right_pt[0] - cx_f), left_pt, right_pt)


def left_right_displacement_from_mask(
    mask: np.ndarray,
    cx: float,
    cy: float,
    angle_rad: float,
) -> tuple[float, float, tuple[float, float] | None, tuple[float, float] | None] | None:
    """L/R displacement from mask (largest contour). Returns (left_disp, right_disp, left_pt, right_pt) or (0,0,None,None)."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0.0, 0.0, None, None)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return (0.0, 0.0, None, None)
    return left_right_displacement_from_contour(contour, cx, cy, angle_rad)


def ellipse_axes_from_mask(mask: np.ndarray) -> tuple[float, float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """
    Fit ellipse to largest contour; return center and major/minor axis endpoints.
    Returns (cx, cy, major_pt1, major_pt2, minor_pt1, minor_pt2) or None.
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
    semi_w, semi_h = width / 2.0, height / 2.0
    if width >= height:
        major_len, minor_len = semi_w, semi_h
        major_angle = angle_rad
        minor_angle = angle_rad + math.pi / 2
    else:
        major_len, minor_len = semi_h, semi_w
        major_angle = angle_rad + math.pi / 2
        minor_angle = angle_rad
    major_pt1 = (cx + major_len * math.cos(major_angle), cy + major_len * math.sin(major_angle))
    major_pt2 = (cx - major_len * math.cos(major_angle), cy - major_len * math.sin(major_angle))
    cx = (major_pt1[0] + major_pt2[0]) / 2.0
    cy = (major_pt1[1] + major_pt2[1]) / 2.0
    minor_pt1 = (cx + minor_len * math.cos(minor_angle), cy + minor_len * math.sin(minor_angle))
    minor_pt2 = (cx - minor_len * math.cos(minor_angle), cy - minor_len * math.sin(minor_angle))
    return (cx, cy, major_pt1, major_pt2, minor_pt1, minor_pt2)


def _line_clip_to_frame(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Clip line through (x1,y1)-(x2,y2) to frame [0,w]x[0,h]. Returns (pt1, pt2) or None."""
    dx, dy = x2 - x1, y2 - y1
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
    """Convert major/minor axis segments to (cx, cy, angle_rad, semi_maj, semi_min)."""
    (x1, y1), (x2, y2) = maj_seg
    (u1, v1), (u2, v2) = min_seg
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = u2 - u1, v2 - v1
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return None
    t = ((u1 - x1) * dy2 - (v1 - y1) * dx2) / denom
    cx, cy = x1 + t * dx1, y1 + t * dy1
    angle = math.atan2(dy1, dx1)
    a = max(math.hypot(x1 - cx, y1 - cy), math.hypot(x2 - cx, y2 - cy))
    b = max(math.hypot(u1 - cx, v1 - cy), math.hypot(u2 - cx, v2 - cy))
    return (cx, cy, angle, a, b)


def ellipse_params_to_segments(
    cx: float, cy: float, angle_rad: float, semi_maj: float, semi_min: float,
    frame_w: int, frame_h: int,
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
    fallback = ((int(cx), int(cy)), (int(cx), int(cy)))
    return (maj_end or fallback, min_end or fallback)


def midpoint_of_segment(ac: tuple[float, float], pc: tuple[float, float]) -> tuple[float, float]:
    """Midpoint of segment AC–PC."""
    return ((ac[0] + pc[0]) / 2.0, (ac[1] + pc[1]) / 2.0)


def major_axis_segment_to_ac_pc_mid(
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """From major axis segment, return AC (min y), PC (max y), and mid."""
    (x1, y1), (x2, y2) = maj_seg
    ac, pc = ((x1, y1), (x2, y2)) if y1 <= y2 else ((x2, y2), (x1, y1))
    return (ac, pc, midpoint_of_segment(ac, pc))


def medial_line_direction_and_half_length(
    ac: tuple[float, float], pc: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Unit direction AC->PC and half-length. Returns (d_unit, half_length)."""
    dx = pc[0] - ac[0]
    dy = pc[1] - ac[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return ((0.0, 0.0), 0.0)
    return ((dx / length, dy / length), length / 2.0)


def ac_pc_from_points_along_medial_line(
    points: np.ndarray,
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """AC/PC as extremes of contour along medial direction. points: (N,2). Returns (AC, PC) or None."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
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
    return (p_min, p_max) if p_min[1] <= p_max[1] else (p_max, p_min)


def ac_pc_from_segment_along_medial_line(
    mask: np.ndarray,
    maj_seg: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """AC/PC from mask contour projected onto medial line. Returns (AC, PC) or None."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
    pts = contour.reshape(-1, 2).astype(np.float64)
    return ac_pc_from_points_along_medial_line(pts, maj_seg)


def line_clip_to_frame(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Clip line through (x1,y1)-(x2,y2) to frame [0,w]x[0,h]. Returns (pt1, pt2) or None. Public alias for drawing."""
    return _line_clip_to_frame(x1, y1, x2, y2, w, h)


def fit_ellipse_to_points(
    points: np.ndarray, frame_w: int, frame_h: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    """Fit ellipse to points; return major/minor axis segments clipped to frame, or None."""
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
    return (maj_end, min_end) if maj_end and min_end else None
