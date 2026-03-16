"""
Displacement estimation module: L/R displacement and area per frame from a medial-line model.

Provides the same analysis used by the GUI (live + save) and usable from the CLI:
- FrameRangeAnalyzer: stateful per-frame analysis (contour buffer, medial line, L/R at configurable position).
- compute_displacement_series: run analyzer over a frame range and return (rows, kinematic_features).

Alongside per-frame segmentation and GAW (glottal area waveform), this adds a displacement time series
and derived kinematic features (open quotient, F0, periodicity, etc.) from L/R or area.
"""

from __future__ import annotations

import collections
import math
from typing import Any

import cv2
import numpy as np

from . import geometry
from . import kinematics


ADAPTIVE_BUFFER_SIZE = 100


class FrameRangeAnalyzer:
    """Stateful per-frame analysis: contour buffer, medial line, L/R displacement. Same logic for GUI and CLI."""

    def __init__(self, beta: float, lr_position: float) -> None:
        self.beta = float(beta)
        self.lr_position = float(lr_position)
        self.reset()

    def set_lr_position(self, lr_position: float) -> None:
        """Update the L/R sampling position (0–1) used for displacement and axes."""
        self.lr_position = float(lr_position)

    def reset(self) -> None:
        self.points_buffer: collections.deque = collections.deque(maxlen=ADAPTIVE_BUFFER_SIZE)
        self.axes_range_start: int | None = None
        self.medial_mid_prev: tuple[float, float] | None = None
        self.medial_d_unit: tuple[float, float] | None = None
        self.medial_half_length: float = 0.0
        self.prev_minor_center: tuple[float, float] | None = None
        self.prev_semi_minor: float = 0.0
        self.prev_disp_left: float | None = None
        self.prev_disp_right: float | None = None
        self.prev_area: float | None = None
        self._last_axes: tuple[Any, Any] | None = None

    def set_initial_from_batch(
        self,
        combined_points: np.ndarray,
        frame_w: int,
        frame_h: int,
        start_frame: int,
    ) -> tuple[Any, Any] | None:
        """Prime state from one-shot accumulated contour (e.g. first 100 frames). Returns axes for display or None."""
        self.points_buffer.clear()
        axes = geometry.fit_ellipse_to_points(combined_points, frame_w, frame_h)
        if axes is None:
            self._last_axes = None
            return None
        maj_seg, min_seg = axes[0], axes[1]
        ac_pc = geometry.ac_pc_from_points_along_medial_line(combined_points, maj_seg)
        if ac_pc is not None:
            ac, pc = ac_pc
            mid = geometry.midpoint_of_segment(ac, pc)
            d_unit, half_length = geometry.medial_line_direction_and_half_length(ac, pc)
        else:
            ac, pc, mid = geometry.major_axis_segment_to_ac_pc_mid(maj_seg)
            d_unit, half_length = geometry.medial_line_direction_and_half_length(ac, pc)
        self.medial_mid_prev = mid
        self.medial_d_unit = d_unit
        self.medial_half_length = half_length
        angle_maj = math.atan2(d_unit[1], d_unit[0])
        params = geometry.segments_to_ellipse_params(maj_seg, min_seg)
        b = params[4] if params is not None else half_length * 0.5
        self.prev_semi_minor = b
        t_axes = max(0.0, min(1.0, self.lr_position))
        cx_axes = mid[0] + (2.0 * t_axes - 1.0) * half_length * d_unit[0]
        cy_axes = mid[1] + (2.0 * t_axes - 1.0) * half_length * d_unit[1]
        self.prev_minor_center = (cx_axes, cy_axes)
        _, min_seg_disp = geometry.ellipse_params_to_segments(
            cx_axes, cy_axes, angle_maj, half_length, b, frame_w, frame_h
        )
        self.axes_range_start = start_frame
        self._last_axes = ((ac, pc), min_seg_disp)
        return self._last_axes

    def process_frame(
        self,
        frame_index: int,
        frame_w: int,
        frame_h: int,
        mask: np.ndarray,
    ) -> tuple[
        float | None, float | None, float,
        tuple[Any, Any] | None,
        tuple[float, float] | None, tuple[float, float] | None,
    ]:
        """
        Update contour buffer and medial state; compute L/R displacement and area.
        Returns (left_disp, right_disp, area, axes_for_display, left_pt, right_pt).
        """
        segmented_pixels = int(np.count_nonzero(mask > 0))
        area_curr = float(segmented_pixels)

        if segmented_pixels >= 30:
            binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) >= 10:
                    self.points_buffer.append(c.astype(np.float32))

        axes_for_display: tuple[Any, Any] | None = None
        if len(self.points_buffer) >= ADAPTIVE_BUFFER_SIZE:
            accumulated = np.vstack(list(self.points_buffer))
            axes_buf = geometry.fit_ellipse_to_points(accumulated, frame_w, frame_h)
            if axes_buf is not None:
                maj_seg_buf, min_seg_buf = axes_buf[0], axes_buf[1]
                ac_pc = geometry.ac_pc_from_points_along_medial_line(accumulated, maj_seg_buf)
                if ac_pc is not None:
                    ac, pc = ac_pc
                    mid = geometry.midpoint_of_segment(ac, pc)
                    d_unit, half_length = geometry.medial_line_direction_and_half_length(ac, pc)
                else:
                    ac, pc, mid = geometry.major_axis_segment_to_ac_pc_mid(maj_seg_buf)
                    d_unit, half_length = geometry.medial_line_direction_and_half_length(ac, pc)
                self.medial_d_unit = d_unit
                self.medial_half_length = half_length
                if min_seg_buf is not None:
                    (mx1, my1), (mx2, my2) = min_seg_buf
                    semi_min_est = 0.5 * math.hypot(mx2 - mx1, my2 - my1)
                else:
                    semi_min_est = half_length * 0.5
                if self.medial_mid_prev is not None:
                    mid_smooth = (
                        (1 - self.beta) * self.medial_mid_prev[0] + self.beta * mid[0],
                        (1 - self.beta) * self.medial_mid_prev[1] + self.beta * mid[1],
                    )
                else:
                    mid_smooth = mid
                self.medial_mid_prev = mid_smooth
                dx, dy = d_unit[0], d_unit[1]
                L = half_length
                t_axes = max(0.0, min(1.0, self.lr_position))
                cx_new = mid_smooth[0] + (2.0 * t_axes - 1.0) * L * dx
                cy_new = mid_smooth[1] + (2.0 * t_axes - 1.0) * L * dy
                if self.prev_minor_center is not None:
                    cx_axes = (1 - self.beta) * self.prev_minor_center[0] + self.beta * cx_new
                    cy_axes = (1 - self.beta) * self.prev_minor_center[1] + self.beta * cy_new
                else:
                    cx_axes, cy_axes = cx_new, cy_new
                self.prev_minor_center = (cx_axes, cy_axes)
                if self.prev_semi_minor > 0:
                    semi_min_smooth = (1 - self.beta) * self.prev_semi_minor + self.beta * semi_min_est
                else:
                    semi_min_smooth = semi_min_est
                self.prev_semi_minor = semi_min_smooth
                angle_maj = math.atan2(dy, dx)
                ac_prev = (mid_smooth[0] - L * dx, mid_smooth[1] - L * dy)
                pc_prev = (mid_smooth[0] + L * dx, mid_smooth[1] + L * dy)
                _, min_seg_disp = geometry.ellipse_params_to_segments(
                    cx_axes, cy_axes, angle_maj, L, semi_min_smooth, frame_w, frame_h
                )
                axes_for_display = ((ac_prev, pc_prev), min_seg_disp)
                self._last_axes = axes_for_display
            else:
                if (
                    self.medial_mid_prev is not None
                    and self.medial_d_unit is not None
                    and self.prev_minor_center is not None
                    and self.prev_semi_minor > 0
                ):
                    dx, dy = self.medial_d_unit[0], self.medial_d_unit[1]
                    L = self.medial_half_length
                    ac_prev = (self.medial_mid_prev[0] - L * dx, self.medial_mid_prev[1] - L * dy)
                    pc_prev = (self.medial_mid_prev[0] + L * dx, self.medial_mid_prev[1] + L * dy)
                    angle_maj = math.atan2(dy, dx)
                    _, min_seg_disp = geometry.ellipse_params_to_segments(
                        self.prev_minor_center[0], self.prev_minor_center[1],
                        angle_maj, L, self.prev_semi_minor, frame_w, frame_h,
                    )
                    axes_for_display = ((ac_prev, pc_prev), min_seg_disp)

        disp_result = None
        if (
            self.medial_d_unit is not None
            and self.medial_half_length > 0
            and self.medial_mid_prev is not None
        ):
            t = max(0.0, min(1.0, self.lr_position))
            dx_u, dy_u = self.medial_d_unit
            L = self.medial_half_length
            cx_disp = self.medial_mid_prev[0] + (2.0 * t - 1.0) * L * dx_u
            cy_disp = self.medial_mid_prev[1] + (2.0 * t - 1.0) * L * dy_u
            angle_disp = math.atan2(dy_u, dx_u)
            disp_result = geometry.left_right_displacement_from_mask(mask, cx_disp, cy_disp, angle_disp)
        elif segmented_pixels >= 30:
            axes_for_disp = geometry.ellipse_axes_from_mask(mask)
            if axes_for_disp is not None:
                cx, cy, maj1, maj2, min1, min2 = axes_for_disp
                angle_maj = math.atan2(maj2[1] - maj1[1], maj2[0] - maj1[0])
                disp_result = geometry.left_right_displacement_from_mask(mask, cx, cy, angle_maj)

        if disp_result is not None:
            self.prev_disp_left, self.prev_disp_right, left_pt, right_pt = disp_result
            self.prev_area = area_curr
            return (self.prev_disp_left, self.prev_disp_right, area_curr, axes_for_display, left_pt, right_pt)
        if self.prev_disp_left is not None and self.prev_disp_right is not None:
            area = float(self.prev_area) if self.prev_area is not None else area_curr
            return (self.prev_disp_left, self.prev_disp_right, area, axes_for_display, None, None)
        return (None, None, area_curr, axes_for_display, None, None)


def compute_displacement_series(
    video_path: str,
    start: int,
    end: int,
    model: Any,
    device: Any,
    *,
    detector: Any | None = None,
    load_frames: Any = None,
    unet_segment_frame: Any = None,
    beta: float = 0.25,
    lr_position: float = 0.5,
    fps: float = 4000.0,
) -> tuple[list[tuple[int, float, float, float]], dict | None]:
    """
    Run displacement analysis over frame range [start, end]; return (rows, kinematic_features).

    Each row is (frame_index, left_disp, right_disp, area). Features are from kinematics.displacement_kinematic_features.

    Requires load_frames(video_path, from_idx, to_idx) -> list[BGR] and unet_segment_frame(gray, model, device) -> mask.
    Optional detector: if given, masks are gated by detector.detect(bgr); None -> zero mask.
    """
    if load_frames is None or unet_segment_frame is None:
        from .metadata import load_frames_bgr_range
        from .utils import unet_segment_frame as _unet_segment
        load_frames = load_frames_bgr_range
        unet_segment_frame = _unet_segment

    analyzer = FrameRangeAnalyzer(beta=beta, lr_position=lr_position)
    rows: list[tuple[int, float, float, float]] = []

    for frame_index in range(start, end + 1):
        frames = load_frames(video_path, frame_index, frame_index)
        if not frames:
            continue
        frm_bgr = frames[0]
        gray = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2GRAY)
        mask = unet_segment_frame(gray, model, device)
        if detector is not None:
            box = detector.detect(frm_bgr)
            if box is None:
                mask[:] = 0
            else:
                x1, y1, x2, y2 = box
                outside = np.ones_like(mask, dtype=bool)
                outside[y1:y2, x1:x2] = False
                mask[outside] = 0
        h, w = frm_bgr.shape[:2]
        left_disp, right_disp, area, _, _, _ = analyzer.process_frame(frame_index, w, h, mask)
        if left_disp is not None and right_disp is not None:
            rows.append((frame_index, float(left_disp), float(right_disp), area))
        elif analyzer.prev_disp_left is not None and analyzer.prev_disp_right is not None:
            a = float(analyzer.prev_area) if analyzer.prev_area is not None else area
            rows.append((frame_index, analyzer.prev_disp_left, analyzer.prev_disp_right, a))

    if not rows:
        return ([], None)
    sel_left = [r[1] for r in rows]
    sel_right = [r[2] for r in rows]
    feats = kinematics.displacement_kinematic_features(sel_left, sel_right, fps)
    return (rows, feats)
