"""qt_app utilities: overlay and drawing. Geometry/kinematics from openglottal."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from openglottal.metadata import (
    load_frames_bgr_range,
    get_video_frame_count,
    get_video_fps,
    load_metadata_for_video,
)
from openglottal import geometry
from openglottal import kinematics

# Re-export for GUI and any local use
from openglottal.kinematics import (
    displacement_kinematic_features,
    kinematic_features_from_opening,
)
from openglottal.geometry import (
    fit_ellipse_to_points,
    segments_to_ellipse_params,
    ellipse_params_to_segments,
    major_axis_segment_to_ac_pc_mid,
    medial_line_direction_and_half_length,
    midpoint_of_segment,
    ac_pc_from_points_along_medial_line,
    ac_pc_from_segment_along_medial_line,
    left_right_displacement_from_mask,
    ellipse_axes_from_mask,
)
_ellipse_axes_from_mask = ellipse_axes_from_mask  # backward compat


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
    axes = geometry.ellipse_axes_from_mask(mask)
    if axes is None:
        return
    cx, cy, maj1, maj2, min1, min2 = axes
    h, w = bgr.shape[:2]
    W, H = w - 1, h - 1
    maj_end = geometry.line_clip_to_frame(maj1[0], maj1[1], maj2[0], maj2[1], W, H)
    min_end = geometry.line_clip_to_frame(min1[0], min1[1], min2[0], min2[1], W, H)
    if maj_end is not None:
        cv2.line(bgr, maj_end[0], maj_end[1], color, thickness)
    if min_end is not None:
        cv2.line(bgr, min_end[0], min_end[1], (255, 0, 0), thickness)  # minor axis in blue
