"""Headless smoke test: Qt MainWindow can be constructed (no display required)."""

from __future__ import annotations

import os
import sys

import pytest

# Offscreen platform works on Linux CI and many macOS setups without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_main_window_smoke() -> None:
    pytest.importorskip("PyQt5.QtWidgets")
    from PyQt5.QtWidgets import QApplication

    from openglottal.qt_app.main import MainWindow

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    win = MainWindow()
    try:
        assert "OpenGlottal" in win.windowTitle()
    finally:
        win.close()
        app.quit()
