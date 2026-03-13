# OpenGlottal Qt GUI

Desktop GUI for viewing high-speed laryngoscopy videos with glottal segmentation overlay, midline/axes, and kinematic metrics.

## Run

From the project root:

```bash
pip install -e ".[gui]"
openglottal-gui
```

Requires **PyQt5** (or PySide6). The `gui` extra installs PyQt5.

## Features

- **Video**: Load a video file; optional `metadata.json` alongside the video for FPS and other keys (default FPS 4000).
- **Frame range**: Start/end frame to restrict playback and inference.
- **Models**: Dropdowns for YOLO detector and U-Net segmenter weights (from `weights/`).
- **Overlay**: Segmentation mask overlay; optional midline (major/minor axes, AC/PC markers) and L/R position (blue line) along the medial line.
- **Waveform**: Left/Right, L−R, or Area (segmentation area per frame); status panel shows Open Quotient, F0, periodicity, etc. from the selected waveform.
- **Playback**: Play/Pause, Step, timeline slider (seeking pauses playback and requests one frame on release).
- **Display crop**: Optional crop of the displayed region (left, top, width, height in pixels).

Changing models or key parameters (L/R position, β, τ, etc.) pauses playback and rebuffers; press Play to resume.
