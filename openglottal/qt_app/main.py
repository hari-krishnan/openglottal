"""Qt GUI: video file, detector/segmenter dropdowns, start/end frame, overlay."""

from __future__ import annotations

import math
import sys
from pathlib import Path
import csv

# Prefer PyQt5; optional PySide6
try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QSpinBox,
        QSlider,
        QFileDialog,
        QGroupBox,
        QProgressBar,
        QFrame,
        QScrollArea,
        QPlainTextEdit,
        QDialog,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
except ImportError:
    try:
        from PySide6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QComboBox,
            QCheckBox,
            QSpinBox,
            QSlider,
            QFileDialog,
            QGroupBox,
            QProgressBar,
            QFrame,
            QScrollArea,
            QPlainTextEdit,
            QDialog,
        )
        from PySide6.QtCore import Qt, QThread, Signal
        from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
        pyqtSignal = Signal
    except ImportError:
        raise ImportError("Install PyQt5 or PySide6: pip install PyQt5")

import collections
import queue
import cv2
import numpy as np
import torch

from openglottal.models import TemporalDetector, UNet
from openglottal.utils import unet_segment_frame
from openglottal.metadata import (
    load_frames_bgr_range,
    get_video_frame_count,
    load_metadata_for_video,
)

from .analyzer import FrameRangeAnalyzer
from .utils import (
    overlay_segment,
    draw_midline_on_bgr,
    draw_axes_on_bgr,
    draw_ac_pc_labels_on_bgr,
    draw_displacement_points_on_bgr,
    displacement_kinematic_features,
    kinematic_features_from_opening,
)


def _find_weights_dir() -> Path:
    """Repo root or cwd; look for weights/."""
    cwd = Path.cwd()
    for d in [cwd, cwd.parent]:
        w = d / "weights"
        if w.is_dir():
            return w
    return cwd / "weights"


def _list_pt_files(dir_path: Path, pattern: str) -> list[str]:
    if not dir_path.is_dir():
        return []
    return sorted(str(p) for p in dir_path.glob(pattern))


class SingleFrameInferenceWorker(QThread):
    """Realtime: infer one frame at a time on request; emit (frame_index, overlay_bgr, mask). Also supports axes from first N frames."""
    frame_ready = pyqtSignal(int, object, object, object)  # index, BGR ndarray, mask ndarray, drift (dx,dy)
    axes_ready = pyqtSignal(object)  # ((maj_pt1, maj_pt2), (min_pt1, min_pt2)) or (None, None)
    displacement_ready = pyqtSignal(int, float, float, float)  # frame_index, left_disp, right_disp, area (pixels)

    def __init__(
        self,
        video_path: str,
        detector_path: str | None,
        unet_path: str | None,
        device: str,
        conf: float = 0.25,
        max_hold_frames: int = 3,
        axes_blend_beta: float = 0.1,
        lr_position: float = 0.5,
    ):
        super().__init__()
        self.video_path = video_path
        self.detector_path = detector_path
        self.unet_path = unet_path
        self.device = device
        self.conf = conf
        self.max_hold_frames = max_hold_frames
        self.axes_blend_beta = axes_blend_beta
        self.lr_position = float(lr_position)
        self._request_queue: queue.Queue[int | None] = queue.Queue()

    def request_frame(self, frame_index: int) -> None:
        self._request_queue.put(frame_index)

    def request_axes(self, start_frame: int, n_frames: int) -> None:
        self._request_queue.put(("axes", start_frame, n_frames))

    def reset_drift(self) -> None:
        """Reset translation drift state (useful when playback loops)."""
        self._request_queue.put(("reset-drift",))

    def set_lr_position(self, lr_position: float) -> None:
        """Update L/R position for subsequent frames without restarting the worker."""
        self._request_queue.put(("set-lr", float(lr_position)))

    def stop(self) -> None:
        self._request_queue.put(None)

    def run(self) -> None:
        import cv2 as _cv2
        device = torch.device(self.device)
        detector = None
        if self.detector_path:
            detector = TemporalDetector(
                self.detector_path,
                conf=self.conf,
                max_hold_frames=self.max_hold_frames,
            )
        model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        model.load_state_dict(
            torch.load(self.unet_path, map_location=device, weights_only=True)
        )
        model.eval()

        analyzer = FrameRangeAnalyzer(
            beta=float(self.axes_blend_beta),
            lr_position=float(self.lr_position),
        )

        while True:
            item = self._request_queue.get()
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "axes":
                _, start_frame, n_frames = item
                all_points = []
                frame_h, frame_w = 0, 0
                for i in range(n_frames):
                    idx = start_frame + i
                    frames = load_frames_bgr_range(self.video_path, idx, idx)
                    if not frames:
                        break
                    frm_bgr = frames[0]
                    frame_h, frame_w = frm_bgr.shape[:2]
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
                    binary = (mask > 0).astype(np.uint8)
                    contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=_cv2.contourArea)
                        if _cv2.contourArea(c) >= 10:
                            all_points.append(c.astype(np.float32))
                if frame_w > 0 and frame_h > 0 and all_points:
                    combined = np.vstack(all_points)
                    axes = analyzer.set_initial_from_batch(combined, frame_w, frame_h, start_frame)
                    self.axes_ready.emit(axes if axes is not None else (None, None))
                else:
                    self.axes_ready.emit((None, None))
                continue
            if isinstance(item, tuple) and item[0] == "reset-drift":
                analyzer.reset()
                continue
            if isinstance(item, tuple) and item[0] == "set-lr":
                _, lr = item
                analyzer.set_lr_position(float(lr))
                continue
            frame_index = item
            frames = load_frames_bgr_range(
                self.video_path, frame_index, frame_index
            )
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
            frame_h, frame_w = frm_bgr.shape[:2]
            left_disp, right_disp, area_curr, axes_for_display, left_pt, right_pt = analyzer.process_frame(
                frame_index, frame_w, frame_h, mask
            )
            overlay = overlay_segment(frm_bgr, mask)
            if axes_for_display is not None:
                self.axes_ready.emit(axes_for_display)
            if left_pt is not None and right_pt is not None:
                draw_displacement_points_on_bgr(overlay, left_pt, right_pt, thickness=1)
            if left_disp is not None and right_disp is not None:
                self.displacement_ready.emit(frame_index, left_disp, right_disp, area_curr)
            elif analyzer.prev_disp_left is not None and analyzer.prev_disp_right is not None:
                area = float(analyzer.prev_area) if analyzer.prev_area is not None else area_curr
                self.displacement_ready.emit(
                    frame_index,
                    analyzer.prev_disp_left,
                    analyzer.prev_disp_right,
                    area,
                )
            self.frame_ready.emit(frame_index, overlay, mask.copy(), (0.0, 0.0))


class SaveAnalysisWorker(QThread):
    """Background worker to recompute displacement/area for a frame range and return waveforms + features."""

    progressed = pyqtSignal(int, int)  # done, total
    finished_ok = pyqtSignal(object, object)  # rows, features
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        detector_path: str | None,
        unet_path: str,
        device_str: str,
        start: int,
        end: int,
        beta: float,
        lr_position: float,
        conf: float,
        max_hold_frames: int,
        fps: float,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detector_path = detector_path
        self.unet_path = unet_path
        self.device_str = device_str
        self.start = start
        self.end = end
        self.beta = float(beta)
        self.lr_position = float(lr_position)
        self.conf = float(conf)
        self.max_hold_frames = int(max_hold_frames)
        self.fps = float(fps)

    def run(self) -> None:
        try:
            device = torch.device(self.device_str)
            detector = None
            if self.detector_path:
                detector = TemporalDetector(
                    self.detector_path,
                    conf=self.conf,
                    max_hold_frames=self.max_hold_frames,
                )
            model = UNet(1, 1, (32, 64, 128, 256)).to(device)
            model.load_state_dict(
                torch.load(self.unet_path, map_location=device, weights_only=True)
            )
            model.eval()
            ADAPTIVE_BUFFER_SIZE = 100
            points_buffer: collections.deque = collections.deque(maxlen=ADAPTIVE_BUFFER_SIZE)
            medial_mid_prev: tuple[float, float] | None = None
            medial_d_unit: tuple[float, float] | None = None
            medial_half_length: float = 0.0
            prev_minor_center: tuple[float, float] | None = None
            prev_semi_minor: float = 0.0
            prev_disp_left: float | None = None
            prev_disp_right: float | None = None
            prev_area: float | None = None
            rows: list[tuple[int, float, float, float]] = []
            total = max(0, self.end - self.start + 1)
            if total == 0:
                self.finished_ok.emit([], None)
                return
            for offset, frame_index in enumerate(range(self.start, self.end + 1), start=1):
                if self.isInterruptionRequested():
                    return
                frames = load_frames_bgr_range(self.video_path, frame_index, frame_index)
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
                frame_h, frame_w = frm_bgr.shape[:2]
                segmented_pixels = int(np.count_nonzero(mask > 0))
                if segmented_pixels >= 30:
                    binary = (mask > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) >= 10:
                            points_buffer.append(c.astype(np.float32))
                if len(points_buffer) >= ADAPTIVE_BUFFER_SIZE:
                    accumulated = np.vstack(list(points_buffer))
                    axes_buf = fit_ellipse_to_points(accumulated, frame_w, frame_h)
                    if axes_buf is not None:
                        maj_seg_buf, min_seg_buf = axes_buf[0], axes_buf[1]
                        ac_pc = ac_pc_from_points_along_medial_line(accumulated, maj_seg_buf)
                        if ac_pc is not None:
                            ac, pc = ac_pc
                            mid = midpoint_of_segment(ac, pc)
                            d_unit, half_length = medial_line_direction_and_half_length(ac, pc)
                            medial_d_unit = d_unit
                            medial_half_length = half_length
                            if min_seg_buf is not None:
                                (mx1, my1), (mx2, my2) = min_seg_buf
                                semi_min_est = 0.5 * math.hypot(mx2 - mx1, my2 - my1)
                            else:
                                semi_min_est = half_length * 0.5
                            if medial_mid_prev is not None:
                                mid_smooth = (
                                    (1 - self.beta) * medial_mid_prev[0] + self.beta * mid[0],
                                    (1 - self.beta) * medial_mid_prev[1] + self.beta * mid[1],
                                )
                            else:
                                mid_smooth = mid
                            medial_mid_prev = mid_smooth
                            dx, dy = d_unit[0], d_unit[1]
                            L = half_length
                            t_axes = max(0.0, min(1.0, self.lr_position))
                            cx_new = mid_smooth[0] + (2.0 * t_axes - 1.0) * L * dx
                            cy_new = mid_smooth[1] + (2.0 * t_axes - 1.0) * L * dy
                            if prev_minor_center is not None:
                                cx_axes = (1 - self.beta) * prev_minor_center[0] + self.beta * cx_new
                                cy_axes = (1 - self.beta) * prev_minor_center[1] + self.beta * cy_new
                            else:
                                cx_axes, cy_axes = cx_new, cy_new
                            prev_minor_center = (cx_axes, cy_axes)
                            if prev_semi_minor > 0:
                                semi_min_smooth = (
                                    (1 - self.beta) * prev_semi_minor + self.beta * semi_min_est
                                )
                            else:
                                semi_min_smooth = semi_min_est
                            prev_semi_minor = semi_min_smooth
                            # angle_maj unused here; medial_d_unit already encodes direction
                disp_result = None
                area_curr = float(segmented_pixels)
                if (
                    medial_d_unit is not None
                    and medial_half_length > 0
                    and medial_mid_prev is not None
                ):
                    t = max(0.0, min(1.0, self.lr_position))
                    dx_u, dy_u = medial_d_unit
                    L = medial_half_length
                    cx_disp = medial_mid_prev[0] + (2.0 * t - 1.0) * L * dx_u
                    cy_disp = medial_mid_prev[1] + (2.0 * t - 1.0) * L * dy_u
                    angle_disp = math.atan2(dy_u, dx_u)
                    disp_result = left_right_displacement_from_mask(
                        mask,
                        cx_disp,
                        cy_disp,
                        angle_disp,
                    )
                elif segmented_pixels >= 30:
                    axes_for_disp = _ellipse_axes_from_mask(mask)
                    if axes_for_disp is not None:
                        cx, cy, maj1, maj2, min1, min2 = axes_for_disp
                        angle_maj = math.atan2(maj2[1] - maj1[1], maj2[0] - maj1[0])
                        disp_result = left_right_displacement_from_mask(
                            mask,
                            cx,
                            cy,
                            angle_maj,
                        )
                if disp_result is not None:
                    prev_disp_left, prev_disp_right, _, _ = disp_result
                    prev_area = area_curr
                    rows.append((frame_index, float(prev_disp_left), float(prev_disp_right), area_curr))
                elif prev_disp_left is not None and prev_disp_right is not None:
                    rows.append(
                        (
                            frame_index,
                            float(prev_disp_left),
                            float(prev_disp_right),
                            float(prev_area) if prev_area is not None else area_curr,
                        )
                    )
                self.progressed.emit(offset, total)
            if not rows:
                self.finished_ok.emit([], None)
                return
            sel_left = [r[1] for r in rows]
            sel_right = [r[2] for r in rows]
            feats = displacement_kinematic_features(sel_left, sel_right, self.fps)
            self.finished_ok.emit(rows, feats)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class DisplacementWaveformWidget(QWidget):
    """Draws displacement vs frame index with selectable modes (Left/Right, L-R, Area)."""
    MAX_POINTS = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = []
        self._left = []
        self._right = []
        self._area = []
        self._mode = "lr"  # "lr" = Left/Right, "diff" = L-R, "area" = max(L-R, 0)
        self.setMinimumHeight(120)
        self.setStyleSheet("background: #1a1a1a;")
        self.setMinimumWidth(200)

    def append(self, frame_index: int, left_disp: float, right_disp: float, area: float) -> None:
        self._frames.append(frame_index)
        self._left.append(left_disp)
        self._right.append(right_disp)
        self._area.append(area)
        if len(self._frames) > self.MAX_POINTS:
            self._frames.pop(0)
            self._left.pop(0)
            self._right.pop(0)
            self._area.pop(0)
        self.update()

    def clear(self) -> None:
        self._frames.clear()
        self._left.clear()
        self._right.clear()
        self._area.clear()
        self.update()

    def get_buffer(self) -> tuple[list[int], list[float], list[float]]:
        """Return (frame_indices, left_disp, right_disp) for analysis."""
        return (list(self._frames), list(self._left), list(self._right))

    def get_area_buffer(self) -> list[float]:
        """Return area (segmentation pixels) per frame in the buffer."""
        return list(self._area)

    def mode(self) -> str:
        """Current display mode: 'lr', 'diff', or 'area'."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Set display mode: 'lr', 'diff', or 'area'."""
        if mode not in {"lr", "diff", "area"}:
            return
        self._mode = mode
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self._frames) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        margin = 4
        plot_w = max(1, w - 2 * margin)
        plot_h = max(1, h - 2 * margin)
        # Choose signal(s) based on mode
        if self._mode == "lr":
            series_left = self._left
            series_right = self._right
            all_vals = self._left + self._right
        elif self._mode == "diff":
            series_diff = [l - r for l, r in zip(self._left, self._right)]
            series_left = series_diff
            series_right = []
            all_vals = series_diff
        else:  # "area": sum of segmentation area (pixels) per frame
            series_area = self._area
            series_left = series_area
            series_right = []
            all_vals = series_area
        if not all_vals:
            return
        lo, hi = min(all_vals), max(all_vals)
        if hi <= lo:
            hi = lo + 1
        n = len(self._frames)
        def to_x(i):
            return margin + (i / max(1, n - 1)) * plot_w
        def to_y(v):
            return margin + plot_h * (1.0 - (v - lo) / (hi - lo))
        # Primary series: red
        painter.setPen(QPen(QColor(255, 100, 100), 1.5))
        if series_left:
            for i in range(1, min(n, len(series_left))):
                painter.drawLine(
                    int(to_x(i - 1)), int(to_y(series_left[i - 1])),
                    int(to_x(i)), int(to_y(series_left[i])),
                )
        # Optional secondary series (Right when in LR mode): blue
        if series_right:
            painter.setPen(QPen(QColor(100, 100, 255), 1.5))
            for i in range(1, min(n, len(series_right))):
                painter.drawLine(
                    int(to_x(i - 1)), int(to_y(series_right[i - 1])),
                    int(to_x(i)), int(to_y(series_right[i])),
                )
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGlottal — Video overlay")
        self.video_path: str | None = None
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        self.worker: SingleFrameInferenceWorker | None = None
        self._playing = False
        self._show_midline = True
        self._global_axes: tuple | None = None  # ((maj_pt1, maj_pt2), (min_pt1, min_pt2)) from first 100 frames
        self._crop_display = False
        self._last_bgr: np.ndarray | None = None
        self._build_ui()
        self._populate_weights()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Left: video display + timeline + playback
        display_widget = QWidget()
        disp_layout = QVBoxLayout()
        display_widget.setLayout(disp_layout)
        disp_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setText("Load a video and run inference.")
        self.frame_label.setStyleSheet("background: #1e1e1e; color: #888;")
        disp_layout.addWidget(self.frame_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self._on_slider_released)
        disp_layout.addWidget(self.slider)
        play_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.clicked.connect(self._on_play_pause_toggle)
        self.btn_play_pause.setEnabled(False)
        play_layout.addWidget(self.btn_play_pause)
        self.btn_step = QPushButton("Step")
        self.btn_step.setToolTip("Advance by one frame")
        self.btn_step.clicked.connect(self._on_step)
        self.btn_step.setEnabled(False)
        play_layout.addWidget(self.btn_step)
        self.btn_midline_toggle = QPushButton("Hide midline")
        self.btn_midline_toggle.setCheckable(True)
        self.btn_midline_toggle.setChecked(True)
        self.btn_midline_toggle.clicked.connect(self._on_midline_toggle)
        play_layout.addWidget(self.btn_midline_toggle)
        play_layout.addWidget(QLabel("L/R pos (0–1):"))
        self.slider_lr_pos = QSlider(Qt.Horizontal)
        self.slider_lr_pos.setMinimum(0)
        self.slider_lr_pos.setMaximum(100)
        self.slider_lr_pos.setValue(50)
        self.slider_lr_pos.setFixedWidth(80)
        self.slider_lr_pos.valueChanged.connect(self._on_lr_pos_changed)
        self.label_lr_pos = QLabel("0.50")
        self.slider_lr_pos.valueChanged.connect(
            lambda v: self.label_lr_pos.setText(f"{v/100:.2f}")
        )
        play_layout.addWidget(self.slider_lr_pos)
        play_layout.addWidget(self.label_lr_pos)
        play_layout.addStretch()
        disp_layout.addLayout(play_layout)
        # Displacement waveform + mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Displacement waveform:"))
        self.combo_waveform_mode = QComboBox()
        self.combo_waveform_mode.addItems(["Left / Right", "L - R", "Area"])
        self.combo_waveform_mode.currentIndexChanged.connect(self._on_waveform_mode_changed)
        mode_row.addWidget(self.combo_waveform_mode)
        self.btn_save_analysis = QPushButton("Save analysis…")
        self.btn_save_analysis.setToolTip(
            "Save kinematic parameters and L/R, L-R, and area waveforms for the current frame range."
        )
        self.btn_save_analysis.clicked.connect(self._on_save_analysis)
        mode_row.addWidget(self.btn_save_analysis)
        mode_row.addStretch()
        disp_layout.addLayout(mode_row)
        self.disp_waveform = DisplacementWaveformWidget()
        self.disp_waveform.set_mode("lr")
        disp_layout.addWidget(self.disp_waveform)
        # Progress bar used when saving analysis; hidden otherwise.
        self.progress_save = QProgressBar()
        self.progress_save.setMinimum(0)
        self.progress_save.setMaximum(100)
        self.progress_save.setValue(0)
        self.progress_save.setTextVisible(True)
        self.progress_save.setFormat("Saving %p%")
        self.progress_save.setFixedHeight(14)
        self.progress_save.setVisible(False)
        disp_layout.addWidget(self.progress_save)
        status_group = QGroupBox("Status (Open Quotient, F0, Periodicity)")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        self.status_label = QLabel("—")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        status_layout.addWidget(self.status_label)
        disp_layout.addWidget(status_group)
        main_layout.addWidget(display_widget, 1)

        # Right: scrollable controls panel
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(320)
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_layout.setAlignment(Qt.AlignTop)

        # Video
        vid_group = QGroupBox("Video")
        vid_layout = QVBoxLayout()
        vid_group.setLayout(vid_layout)
        self.video_label = QLabel("No video loaded")
        self.video_label.setWordWrap(True)
        vid_layout.addWidget(self.video_label)
        self.btn_open = QPushButton("Open video…")
        self.btn_open.clicked.connect(self._open_video)
        vid_layout.addWidget(self.btn_open)
        vid_layout.addWidget(QLabel("FPS (Hz):"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setMinimum(1)
        self.spin_fps.setMaximum(99999)
        self.spin_fps.setValue(4000)
        self.spin_fps.setToolTip("Frame rate (Hz) for F0 etc. Default 4000; set from metadata.json if enabled.")
        vid_layout.addWidget(self.spin_fps)
        self.check_metadata = QCheckBox("Use metadata.json (if present)")
        self.check_metadata.setChecked(True)
        vid_layout.addWidget(self.check_metadata)
        self.meta_text = QPlainTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(40)  # ~1–2 lines visible, scroll for more
        self.meta_text.setPlaceholderText("Metadata: none")
        vid_layout.addWidget(self.meta_text)
        controls_layout.addWidget(vid_group)

        # Frame range
        range_group = QGroupBox("Frame range (crop in time)")
        range_layout = QHBoxLayout()
        range_group.setLayout(range_layout)
        range_layout.addWidget(QLabel("Start:"))
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(0)
        self.spin_start.setMaximum(999999)
        self.spin_start.valueChanged.connect(self._clamp_end)
        self.spin_start.valueChanged.connect(self._update_slider_range)
        self.spin_start.valueChanged.connect(self._update_play_button_state)
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        self.spin_end = QSpinBox()
        self.spin_end.setMinimum(0)
        self.spin_end.setMaximum(999999)
        self.spin_end.valueChanged.connect(self._clamp_start)
        self.spin_end.valueChanged.connect(self._update_slider_range)
        self.spin_end.valueChanged.connect(self._update_play_button_state)
        range_layout.addWidget(self.spin_end)
        range_layout.addStretch()
        controls_layout.addWidget(range_group)

        # Display crop (purely for display; does not affect processing)
        crop_group = QGroupBox("Display crop")
        crop_layout = QVBoxLayout()
        crop_group.setLayout(crop_layout)
        self.check_crop = QCheckBox("Crop display")
        self.check_crop.setChecked(False)
        self.check_crop.stateChanged.connect(self._on_crop_toggled)
        crop_layout.addWidget(self.check_crop)
        crop_row1 = QHBoxLayout()
        crop_row1.addWidget(QLabel("L:"))
        self.spin_crop_left = QSpinBox()
        self.spin_crop_left.setMinimum(0)
        self.spin_crop_left.setMaximum(9999)
        self.spin_crop_left.setValue(0)
        self.spin_crop_left.valueChanged.connect(self._on_crop_changed)
        crop_row1.addWidget(self.spin_crop_left)
        crop_row1.addWidget(QLabel("T:"))
        self.spin_crop_top = QSpinBox()
        self.spin_crop_top.setMinimum(0)
        self.spin_crop_top.setMaximum(9999)
        self.spin_crop_top.setValue(0)
        self.spin_crop_top.valueChanged.connect(self._on_crop_changed)
        crop_row1.addWidget(self.spin_crop_top)
        crop_layout.addLayout(crop_row1)
        crop_row2 = QHBoxLayout()
        crop_row2.addWidget(QLabel("W:"))
        self.spin_crop_width = QSpinBox()
        self.spin_crop_width.setMinimum(1)
        self.spin_crop_width.setMaximum(9999)
        self.spin_crop_width.setValue(640)
        self.spin_crop_width.valueChanged.connect(self._on_crop_changed)
        crop_row2.addWidget(self.spin_crop_width)
        crop_row2.addWidget(QLabel("H:"))
        self.spin_crop_height = QSpinBox()
        self.spin_crop_height.setMinimum(1)
        self.spin_crop_height.setMaximum(9999)
        self.spin_crop_height.setValue(480)
        self.spin_crop_height.valueChanged.connect(self._on_crop_changed)
        crop_row2.addWidget(self.spin_crop_height)
        crop_layout.addLayout(crop_row2)
        controls_layout.addWidget(crop_group)

        # Models
        model_group = QGroupBox("Models")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        model_layout.addWidget(QLabel("Detector (YOLO .pt):"))
        self.combo_detector = QComboBox()
        self.combo_detector.setEditable(True)
        self.combo_detector.currentIndexChanged.connect(self._on_model_changed)
        self.combo_detector.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.combo_detector)
        model_layout.addWidget(QLabel("Segmenter (U-Net .pt):"))
        self.combo_unet = QComboBox()
        self.combo_unet.setEditable(True)
        self.combo_unet.currentIndexChanged.connect(self._on_model_changed)
        self.combo_unet.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.combo_unet)
        model_layout.addWidget(QLabel("Axes learning β:"))
        beta_layout = QHBoxLayout()
        self.slider_beta = QSlider(Qt.Horizontal)
        self.slider_beta.setMinimum(1)
        self.slider_beta.setMaximum(50)
        self.slider_beta.setValue(25)
        self.slider_beta.valueChanged.connect(self._on_model_changed)
        self.label_beta = QLabel("0.25")
        self.slider_beta.valueChanged.connect(lambda v: self.label_beta.setText(f"{v/100:.2f}"))
        beta_layout.addWidget(self.slider_beta)
        beta_layout.addWidget(self.label_beta)
        model_layout.addLayout(beta_layout)
        model_layout.addWidget(QLabel("τ (detector threshold):"))
        tau_layout = QHBoxLayout()
        self.slider_tau = QSlider(Qt.Horizontal)
        self.slider_tau.setMinimum(1)
        self.slider_tau.setMaximum(100)
        self.slider_tau.setValue(25)
        self.slider_tau.valueChanged.connect(self._on_model_changed)
        self.label_tau = QLabel("0.25")
        self.slider_tau.valueChanged.connect(lambda v: self.label_tau.setText(f"{v/100:.2f}"))
        tau_layout.addWidget(self.slider_tau)
        tau_layout.addWidget(self.label_tau)
        model_layout.addLayout(tau_layout)
        model_layout.addWidget(QLabel("Hold (frames):"))
        hold_layout = QHBoxLayout()
        self.slider_hold = QSlider(Qt.Horizontal)
        self.slider_hold.setMinimum(1)
        self.slider_hold.setMaximum(20)
        self.slider_hold.setValue(3)
        self.slider_hold.valueChanged.connect(self._on_model_changed)
        self.label_hold = QLabel("3")
        self.slider_hold.valueChanged.connect(lambda v: self.label_hold.setText(str(v)))
        hold_layout.addWidget(self.slider_hold)
        hold_layout.addWidget(self.label_hold)
        model_layout.addLayout(hold_layout)
        controls_layout.addWidget(model_group)

        scroll = QScrollArea()
        scroll.setWidget(controls_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumWidth(324)
        main_layout.addWidget(scroll)

    def _populate_weights(self):
        wdir = _find_weights_dir()
        yolo_paths = _list_pt_files(wdir, "*.pt")
        for p in yolo_paths:
            name = Path(p).name
            if "yolo" in name.lower():
                self.combo_detector.addItem(name, p)
            if "unet" in name.lower():
                self.combo_unet.addItem(name, p)
        if self.combo_detector.count():
            self.combo_detector.setCurrentIndex(0)
        if self.combo_unet.count():
            self.combo_unet.setCurrentIndex(0)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video (*.avi *.mp4 *.mov);;All (*)",
        )
        if not path:
            return
        self._playing = False
        self.btn_play_pause.setText("Play")
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.worker = None
        self.video_path = path
        self.total_frames = get_video_frame_count(path)
        meta: dict = load_metadata_for_video(path) if self.check_metadata.isChecked() else {}
        fps_from_meta = meta.get("fps") or meta.get("frame_rate") or meta.get("frame rate")
        # FPS: only from metadata.json when enabled; never from AVI. Default 4000.
        if fps_from_meta is not None:
            try:
                fps_val = float(fps_from_meta)
                if fps_val >= 1:
                    self.spin_fps.setValue(int(round(fps_val)))
            except (TypeError, ValueError):
                pass
        # Update video + metadata labels
        self.video_label.setText(f"{Path(path).name} — {self.total_frames} frames")
        if meta:
            lines: list[str] = []
            if fps_from_meta is not None:
                lines.append(f"fps = {fps_from_meta}")
            for k, v in meta.items():
                if k in {"fps", "frame_rate", "frame rate"}:
                    continue
                lines.append(f"{k} = {v}")
            self.meta_text.setPlainText("\n".join(str(line) for line in lines))
        else:
            self.meta_text.clear()
        self.spin_start.setMaximum(max(0, self.total_frames - 1))
        self.spin_end.setMaximum(max(0, self.total_frames - 1))
        self.spin_start.setValue(0)
        self.spin_end.setValue(max(0, self.total_frames - 1))
        self._global_axes = None
        self._last_bgr = None
        self.disp_waveform.clear()
        self.status_label.setText("—")
        self._update_slider_range()
        self._update_play_button_state()
        self._show_current_frame_or_placeholder()

    def _clamp_end(self):
        s = self.spin_start.value()
        if self.spin_end.value() < s:
            self.spin_end.setValue(s)

    def _clamp_start(self):
        e = self.spin_end.value()
        if self.spin_start.value() > e:
            self.spin_start.setValue(e)

    def _update_slider_range(self) -> None:
        start = self.spin_start.value()
        end = self.spin_end.value()
        n = max(0, end - start + 1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, n - 1))
        if self.slider.value() > self.slider.maximum():
            self.slider.setValue(self.slider.maximum())

    def _current_frame_index(self) -> int:
        start = self.spin_start.value()
        return start + self.slider.value()

    def _on_model_changed(self) -> None:
        """When a param (L/R pos, beta, tau, models, etc.) changes: pause and stop worker. On next Play we rebuffer from scratch (initial 100-frame axes, then 100-frame buffer refills from current position)."""
        self._global_axes = None
        if self.worker and self.worker.isRunning():
            self._playing = False
            self.btn_play_pause.setText("Play")
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
            self.status_label.setText("Paused — params changed. Press Play to resume; axes and 100-frame buffer will recompute.")
        self._update_play_button_state()

    def _on_lr_pos_changed(self, value: int) -> None:
        """While adjusting L/R slider: pause playback and repaint current frame with new line position."""
        # Pause playback but keep worker alive so we can reuse the buffered state.
        if self._playing:
            self._playing = False
            self.btn_play_pause.setText("Play")
        lr = float(value) / 100.0
        # Inform worker of new L/R position and ask it to recompute the current frame.
        if self.worker and self.worker.isRunning():
            self.worker.set_lr_position(lr)
            self.worker.request_frame(self._current_frame_index())

    def _update_play_button_state(self) -> None:
        unet_path = self._get_unet_path()
        has_range = self.video_path and (self.spin_end.value() - self.spin_start.value() >= 0)
        enabled = bool(self.video_path and unet_path and has_range)
        self.btn_play_pause.setEnabled(enabled)
        self.btn_step.setEnabled(bool(self.video_path and has_range))

    def _get_detector_path(self) -> str | None:
        idx = self.combo_detector.currentIndex()
        if idx >= 0 and self.combo_detector.currentData():
            return self.combo_detector.currentData()
        return self.combo_detector.currentText().strip() or None

    def _get_unet_path(self) -> str | None:
        idx = self.combo_unet.currentIndex()
        if idx >= 0 and self.combo_unet.currentData():
            return self.combo_unet.currentData()
        return self.combo_unet.currentText().strip() or None

    def _start_worker(self) -> None:
        if self.worker is not None:
            return
        unet_path = self._get_unet_path()
        if not unet_path or not self.video_path:
            return
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        conf = self.slider_tau.value() / 100.0
        max_hold = self.slider_hold.value()
        axes_blend_beta = self.slider_beta.value() / 100.0
        lr_position = self.slider_lr_pos.value() / 100.0
        self.worker = SingleFrameInferenceWorker(
            self.video_path,
            self._get_detector_path(),
            unet_path,
            device,
            conf=conf,
            max_hold_frames=max_hold,
            axes_blend_beta=axes_blend_beta,
            lr_position=lr_position,
        )
        self.worker.frame_ready.connect(self._on_single_frame_ready)
        self.worker.axes_ready.connect(self._on_axes_ready)
        self.worker.displacement_ready.connect(self._on_displacement_ready)
        self.worker.start()
        self._request_global_axes()

    def _request_global_axes(self) -> None:
        """(Re)compute global axes from the first 100 frames of the range. New worker starts with empty buffer; 100-frame sliding buffer refills as frames are requested."""
        if not self.worker or not self.video_path:
            return
        start = self.spin_start.value()
        end = self.spin_end.value()
        n = min(100, max(0, end - start + 1))
        if n >= 5:
            self.worker.request_axes(start, n)

    def _show_bgr(self, bgr: np.ndarray) -> None:
        """Display a BGR image in the frame label; apply display crop if enabled."""
        self._last_bgr = bgr.copy()
        h, w = bgr.shape[:2]
        # Clamp crop spinbox max to current frame size
        self.spin_crop_left.setMaximum(max(0, w - 1))
        self.spin_crop_top.setMaximum(max(0, h - 1))
        self.spin_crop_width.setMaximum(w)
        self.spin_crop_height.setMaximum(h)
        if self._crop_display:
            left = max(0, min(self.spin_crop_left.value(), w - 1))
            top = max(0, min(self.spin_crop_top.value(), h - 1))
            cw = max(1, min(self.spin_crop_width.value(), w - left))
            ch = max(1, min(self.spin_crop_height.value(), h - top))
            bgr = bgr[top : top + ch, left : left + cw]
            h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _load_and_show_raw_frame(self, frame_index: int) -> None:
        """Load the video frame at frame_index and show it (no overlay)."""
        if not self.video_path:
            return
        frames = load_frames_bgr_range(self.video_path, frame_index, frame_index)
        if frames:
            self._show_bgr(frames[0])

    def _ensure_frame(self, frame_index: int) -> None:
        """Show current raw frame immediately, then request inference (no cache)."""
        self._load_and_show_raw_frame(frame_index)
        if not self._get_unet_path():
            return
        self._start_worker()
        if self.worker:
            self.worker.request_frame(frame_index)

    def _on_axes_ready(self, axes: tuple) -> None:
        self._global_axes = axes

    def _on_displacement_ready(self, frame_index: int, left_disp: float, right_disp: float, area: float) -> None:
        if not self._playing and frame_index != self._current_frame_index():
            return
        self.disp_waveform.append(frame_index, left_disp, right_disp, area)
        self._update_status_from_buffer()

    def _update_status_from_buffer(self) -> None:
        """Run kinematic analysis on waveform buffer (right inverted; negative = closed) and update status box."""
        frames, left, right = self.disp_waveform.get_buffer()
        if not left or not right:
            self.status_label.setText("—")
            return
        fps = float(self.spin_fps.value())
        mode = self.disp_waveform.mode()
        if mode == "lr":
            # Use full left/right information (same as before)
            feats = displacement_kinematic_features(left, right, fps)
        elif mode == "diff":
            # Use opening = max(L - R, 0) as the signal for features
            opening = [max(l - r, 0.0) for l, r in zip(left, right)]
            feats = kinematic_features_from_opening(opening, fps)
        else:  # "area": use segmentation area (pixels) as opening
            area = self.disp_waveform.get_area_buffer()
            feats = kinematic_features_from_opening(area, fps)
        if feats is None:
            self.status_label.setText("—")
            return
        oq = feats.get("open_quotient")
        f0_hz = feats.get("f0_hz")
        periodicity = feats.get("periodicity")
        cv = feats.get("cv")
        parts = []
        if oq is not None:
            parts.append(f"Open Quotient: {oq:.3f}")
        if f0_hz is not None:
            fps_used = feats.get("fps_used")
            parts.append(f"F0: {f0_hz:.1f} Hz ({fps_used:.0f} fps)" if fps_used is not None else f"F0: {f0_hz:.1f} Hz")
        if periodicity is not None:
            parts.append(f"Periodicity: {periodicity:.3f}")
        if cv is not None:
            parts.append(f"CV: {cv:.3f}")
        self.status_label.setText("  |  ".join(parts) if parts else "—")

    def _on_single_frame_ready(self, index: int, bgr: np.ndarray, mask: np.ndarray, drift_xy: tuple[float, float]) -> None:
        """Show overlay; optionally draw axes. When playing, advance slider and request next; when paused, only update if this is the requested frame (avoids jump from stale signals after param change)."""
        # When paused, ignore stale frame_ready from a worker we just stopped (would move display "back and forth")
        if not self._playing:
            if index != self._current_frame_index():
                return
        if self._show_midline and self._global_axes is not None:
            draw_axes_on_bgr(
                bgr,
                self._global_axes,
                offset_xy=drift_xy,
                color_major=(0, 0, 255),
                color_minor=(255, 0, 0),
                thickness=1,
            )
            maj_seg = self._global_axes[0]
            if maj_seg is not None:
                draw_ac_pc_labels_on_bgr(bgr, maj_seg, mask=None, offset_xy=drift_xy, thickness=1)
        self._show_bgr(bgr)
        if not self._playing:
            return
        start = self.spin_start.value()
        end = self.spin_end.value()
        self.slider.blockSignals(True)
        self.slider.setValue(index - start)
        self.slider.blockSignals(False)
        next_index = index + 1
        if next_index > end:
            # Loop back: recompute global axes and reset drift
            next_index = start
            self._global_axes = None
            self.disp_waveform.clear()
            self.status_label.setText("—")
            if self.worker:
                self.worker.reset_drift()
            self._request_global_axes()
        if self.worker:
            self.worker.request_frame(next_index)

    def _on_slider(self, pos: int) -> None:
        """User is dragging the timeline: pause playback and show raw frame only (no inference request)."""
        if not self.video_path:
            return
        if self._playing:
            self._playing = False
            self.btn_play_pause.setText("Play")
        self._load_and_show_raw_frame(self._current_frame_index())

    def _on_slider_released(self) -> None:
        """User released the timeline slider: request one frame for overlay at current position."""
        if not self.video_path:
            return
        if not self._get_unet_path():
            return
        self._start_worker()
        if self.worker:
            self.worker.request_frame(self._current_frame_index())

    def _on_step(self) -> None:
        """Advance by one frame."""
        if not self.video_path or self.slider.value() >= self.slider.maximum():
            return
        self.slider.setValue(self.slider.value() + 1)

    def _on_save_analysis(self) -> None:
        """Save kinematic parameters and waveforms for current frame range to a CSV file.

        Recomputes displacement and area for each frame from Start to End in order,
        using the current model paths and parameter sliders. No images are rendered.
        """
        if not self.video_path:
            return
        unet_path = self._get_unet_path()
        if not unet_path:
            return
        start = self.spin_start.value()
        end = self.spin_end.value()
        if end < start:
            return
        # Pause playback while analysis runs
        if self._playing:
            self._playing = False
            self.btn_play_pause.setText("Play")
        # Device and detector settings consistent with _start_worker
        device_str = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        device = torch.device(device_str)
        conf = self.slider_tau.value() / 100.0
        max_hold = self.slider_hold.value()
        det_path = self._get_detector_path()
        detector = None
        if det_path:
            detector = TemporalDetector(det_path, conf=conf, max_hold_frames=max_hold)
        model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        model.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
        model.eval()
        # Axes / medial line state mirroring worker logic
        ADAPTIVE_BUFFER_SIZE = 100
        points_buffer: collections.deque = collections.deque(maxlen=ADAPTIVE_BUFFER_SIZE)
        medial_mid_prev: tuple[float, float] | None = None
        medial_d_unit: tuple[float, float] | None = None
        medial_half_length: float = 0.0
        prev_minor_center: tuple[float, float] | None = None
        prev_semi_minor: float = 0.0
        prev_disp_left: float | None = None
        prev_disp_right: float | None = None
        prev_area: float | None = None
        prev_adaptive_ellipse: tuple[float, float, float, float, float] | None = None
        beta = float(self.slider_beta.value() / 100.0)
        lr_position = float(self.slider_lr_pos.value() / 100.0)
        # Prepare output container
        rows: list[tuple[int, float, float, float]] = []
        n_frames = end - start + 1
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save analysis",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        # Progress bar over frames (compute) + quick bump during write
        self.progress_save.setVisible(True)
        self.progress_save.setMinimum(0)
        self.progress_save.setMaximum(n_frames)
        self.progress_save.setValue(0)
        try:
            for offset, frame_index in enumerate(range(start, end + 1), start=1):
                frames = load_frames_bgr_range(self.video_path, frame_index, frame_index)
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
                frame_h, frame_w = frm_bgr.shape[:2]
                segmented_pixels = int(np.count_nonzero(mask > 0))
                # Update contour buffer for medial line / axes if there is a usable mask
                if segmented_pixels >= 30:
                    binary = (mask > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) >= 10:
                            points_buffer.append(c.astype(np.float32))
                if len(points_buffer) >= ADAPTIVE_BUFFER_SIZE:
                    accumulated = np.vstack(list(points_buffer))
                    axes_buf = fit_ellipse_to_points(accumulated, frame_w, frame_h)
                    if axes_buf is not None:
                        maj_seg_buf, min_seg_buf = axes_buf[0], axes_buf[1]
                        ac_pc = ac_pc_from_points_along_medial_line(accumulated, maj_seg_buf)
                        if ac_pc is not None:
                            ac, pc = ac_pc
                            mid = midpoint_of_segment(ac, pc)
                            d_unit, half_length = medial_line_direction_and_half_length(ac, pc)
                            medial_d_unit = d_unit
                            medial_half_length = half_length
                            if min_seg_buf is not None:
                                (mx1, my1), (mx2, my2) = min_seg_buf
                                semi_min_est = 0.5 * math.hypot(mx2 - mx1, my2 - my1)
                            else:
                                semi_min_est = half_length * 0.5
                            if medial_mid_prev is not None:
                                mid_smooth = (
                                    (1 - beta) * medial_mid_prev[0] + beta * mid[0],
                                    (1 - beta) * medial_mid_prev[1] + beta * mid[1],
                                )
                            else:
                                mid_smooth = mid
                            medial_mid_prev = mid_smooth
                            dx, dy = d_unit[0], d_unit[1]
                            L = half_length
                            # Smooth blue-line center and semi-minor
                            t_axes = max(0.0, min(1.0, lr_position))
                            cx_new = mid_smooth[0] + (2.0 * t_axes - 1.0) * L * dx
                            cy_new = mid_smooth[1] + (2.0 * t_axes - 1.0) * L * dy
                            if prev_minor_center is not None:
                                cx_axes = (1 - beta) * prev_minor_center[0] + beta * cx_new
                                cy_axes = (1 - beta) * prev_minor_center[1] + beta * cy_new
                            else:
                                cx_axes, cy_axes = cx_new, cy_new
                            prev_minor_center = (cx_axes, cy_axes)
                            if prev_semi_minor > 0:
                                semi_min_smooth = (1 - beta) * prev_semi_minor + beta * semi_min_est
                            else:
                                semi_min_smooth = semi_min_est
                            prev_semi_minor = semi_min_smooth
                            angle_maj = math.atan2(dy, dx)
                            prev_adaptive_ellipse = (
                                mid_smooth[0],
                                mid_smooth[1],
                                angle_maj,
                                L,
                                semi_min_smooth,
                            )
                # Displacement at current frame, using buffered medial line when available
                disp_result = None
                area_curr = float(segmented_pixels)
                if (
                    medial_d_unit is not None
                    and medial_half_length > 0
                    and medial_mid_prev is not None
                ):
                    t = max(0.0, min(1.0, lr_position))
                    dx_u, dy_u = medial_d_unit
                    L = medial_half_length
                    cx_disp = medial_mid_prev[0] + (2.0 * t - 1.0) * L * dx_u
                    cy_disp = medial_mid_prev[1] + (2.0 * t - 1.0) * L * dy_u
                    angle_disp = math.atan2(dy_u, dx_u)
                    disp_result = left_right_displacement_from_mask(
                        mask,
                        cx_disp,
                        cy_disp,
                        angle_disp,
                    )
                elif segmented_pixels >= 30:
                    axes_for_disp = _ellipse_axes_from_mask(mask)
                    if axes_for_disp is not None:
                        cx, cy, maj1, maj2, min1, min2 = axes_for_disp
                        angle_maj = math.atan2(maj2[1] - maj1[1], maj2[0] - maj1[0])
                        disp_result = left_right_displacement_from_mask(
                            mask,
                            cx,
                            cy,
                            angle_maj,
                        )
                if disp_result is not None:
                    prev_disp_left, prev_disp_right, _, _ = disp_result
                    prev_area = area_curr
                    rows.append((frame_index, float(prev_disp_left), float(prev_disp_right), area_curr))
                elif prev_disp_left is not None and prev_disp_right is not None:
                    rows.append(
                        (
                            frame_index,
                            float(prev_disp_left),
                            float(prev_disp_right),
                            float(prev_area) if prev_area is not None else area_curr,
                        )
                    )
                self.progress_save.setValue(offset)
            if not rows:
                return
            sel_left = [r[1] for r in rows]
            sel_right = [r[2] for r in rows]
            fps = float(self.spin_fps.value())
            lr_position = float(self.slider_lr_pos.value() / 100.0)
            feats = displacement_kinematic_features(sel_left, sel_right, fps)
            # Quick write phase (fast relative to compute)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                if feats:
                    writer.writerow(["# open_quotient", feats.get("open_quotient")])
                    writer.writerow(["# f0_hz", feats.get("f0_hz")])
                    writer.writerow(["# periodicity", feats.get("periodicity")])
                    writer.writerow(["# cv", feats.get("cv")])
                    writer.writerow(["# fps_used", feats.get("fps_used")])
                    writer.writerow(["# mean_opening", feats.get("mean_opening")])
                    writer.writerow(["# std_opening", feats.get("std_opening")])
                writer.writerow(["# lr_position_0_1", lr_position])
                writer.writerow(
                    ["frame", "left_disp", "right_disp", "diff_L_minus_R", "area", "lr_position_0_1"]
                )
                for f_idx, l, r, a in rows:
                    writer.writerow([f_idx, l, r, l - r, a, lr_position])
        finally:
            self.progress_save.setVisible(False)

    def _on_crop_toggled(self) -> None:
        self._crop_display = self.check_crop.isChecked()
        if self._last_bgr is not None:
            self._show_bgr(self._last_bgr)

    def _on_crop_changed(self) -> None:
        if self._crop_display and self._last_bgr is not None:
            self._show_bgr(self._last_bgr)

    def _show_current_frame_or_placeholder(self) -> None:
        if not self.video_path:
            self.frame_label.setText("Load a video.")
            return
        start = self.spin_start.value()
        end = self.spin_end.value()
        if end < start:
            self.frame_label.setText("Set frame range.")
            return
        frame_idx = self._current_frame_index()
        self._load_and_show_raw_frame(frame_idx)
        if self._get_unet_path():
            self._start_worker()
            if self.worker:
                self.worker.request_frame(frame_idx)

    def _on_midline_toggle(self) -> None:
        self._show_midline = self.btn_midline_toggle.isChecked()
        self.btn_midline_toggle.setText("Hide midline" if self._show_midline else "Show midline")

    def _on_waveform_mode_changed(self, index: int) -> None:
        """Switch displacement waveform display between Left/Right, L-R, and Area. Display-only; do not request a frame (avoids queue reorder and jitter)."""
        if index == 0:
            mode = "lr"
        elif index == 1:
            mode = "diff"
        else:
            mode = "area"
        self.disp_waveform.set_mode(mode)
        self._update_status_from_buffer()

    def _on_play_pause_toggle(self) -> None:
        if self._playing:
            self._playing = False
            self.btn_play_pause.setText("Play")
            return
        if not self.video_path or not self._get_unet_path():
            return
        start = self.spin_start.value()
        end = self.spin_end.value()
        if start > end:
            return
        self._playing = True
        self.btn_play_pause.setText("Pause")
        self._start_worker()
        frame_index = self._current_frame_index()
        self._load_and_show_raw_frame(frame_index)
        if self.worker:
            self.worker.request_frame(frame_index)

    def closeEvent(self, event):
        self._playing = False
        self.btn_play_pause.setText("Play")
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        event.accept()


def run_gui():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    run_gui()
