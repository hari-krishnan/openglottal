# OpenGlottal

![Patient 1 montage](paper/patient1_montage.png)

Open-source toolkit for automated glottal area segmentation from high-speed videoendoscopy (HSV).

OpenGlottal combines a YOLOv8 glottis detector, a U-Net pixel-level segmenter, and a temporal vocal fold tracker into a single, reproducible inference and training pipeline — trained and evaluated on the [GIRAFE dataset](https://zenodo.org/records/13773163) ([dataset paper](https://doi.org/10.1016/j.dib.2024.111376)) and [BAGLS](https://zenodo.org/record/3381469) ([Scientific Data, 2020](https://doi.org/10.1038/s41597-020-0526-3)).

---

## Pipelines

Four pipelines are provided (three YOLO-gated and one U-Net-only), reflecting a progression in complexity and accuracy:

| Pipeline | Flag | Description |
|----------|------|-------------|
| **1 — VFT** | `vft` | YOLO detects the glottis → fixed-size crop → motion-based VocalFoldTracker |
| **2 — Guided VFT** | `guided-vft` | YOLO bbox used as ROI mask on the full frame → YOLOGuidedVFT |
| **3 — U-Net** | `unet` | YOLO + U-Net: full-frame U-Net → pixel count restricted to YOLO bbox (detection-gated) |
| **4 — U-Net only** | `unet-only` | Full-frame U-Net only, no YOLO gate (requires `--unet-weights` only) |

All pipelines produce a per-frame **glottal area waveform** from which kinematic features (open quotient, fundamental frequency, periodicity, etc.) are extracted for downstream clinical analysis.

---

## Installation

Install from source (clone the repo first):

```bash
git clone https://github.com/hari-krishnan/openglottal.git
cd openglottal
pip install -e ".[dev]"
```

Or, from a local clone:

```bash
cd /path/to/openglottal
pip install -e ".[dev]"
```

*(A `pip install openglottal` option will work once the package is published to PyPI.)*

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Ultralytics ≥ 8.0, OpenCV ≥ 4.8

**Weights:** Pre-trained YOLO and U-Net weights are in `weights/`: `openglottal_yolo.pt` and `openglottal_unet.pt`. To train your own, see [Training](#training).

---

## Quick Start

### Python API

```python
import torch
from openglottal import TemporalDetector, UNet, extract_features_unet

device = torch.device("mps")   # or "cuda" / "cpu"

detector = TemporalDetector("weights/openglottal_yolo.pt")

model = UNet(1, 1, (32, 64, 128, 256)).to(device)
model.load_state_dict(torch.load("weights/openglottal_unet.pt", map_location=device))
model.eval()

features = extract_features_unet("video.avi", detector, model, device)
print(features)
# {'area_mean': 312.4, 'area_std': 98.1, 'open_quotient': 0.61, 'f0': 0.017, ...}
```

### CLI

```bash
# U-Net pipeline (recommended)
openglottal run video.avi \
    --yolo-weights weights/openglottal_yolo.pt \
    --unet-weights weights/openglottal_unet.pt \
    --pipeline unet \
    --output results/

# Motion-based pipeline (no U-Net weights needed)
openglottal run video.avi \
    --yolo-weights weights/openglottal_yolo.pt \
    --pipeline guided-vft \
    --output results/
```

---

## Evaluation

Compare against the GIRAFE paper baselines on the 4 held-out test patients (80 frames total):

```bash
python scripts/eval_girafe.py \
    --images-dir  GIRAFE/Training/imagesTr \
    --labels-dir  GIRAFE/Training/labelsTr \
    --training-json GIRAFE/Training/training.json \
    --unet-weights weights/openglottal_unet.pt \
    --yolo-weights weights/openglottal_yolo.pt \
    --device mps
```

Results are printed alongside the published GIRAFE baselines for direct comparison.  Pass `--output-json results.json` to save raw per-frame scores.

### Results (GIRAFE test split, 4 patients, 80 frames)

| Method | Det.Recall | Dice | IoU | Dice≥0.5 |
|--------|-----------|------|-----|----------|
| InP (GIRAFE paper) | n/a | 0.713 | n/a | n/a |
| U-Net (GIRAFE paper) | n/a | 0.643 | n/a | n/a |
| SwinUNetV2 (paper) | n/a | 0.621 | n/a | n/a |
| **U-Net only** | n/a | **0.809** | **0.699** | **96.2%** |
| YOLO+OTSU | 0.95 | 0.230 | 0.136 | 2.5% |
| YOLO+UNet | 0.95 | 0.746 | 0.629 | 83.8% |
| YOLO-Crop+UNet† | 0.95 | 0.697 | 0.567 | 77.5% |
| YOLO+Motion | 0.95 | 0.349 | 0.234 | 23.5% |

- **Det.Recall** — fraction of frames where YOLO detected a glottis
- **Dice** — mean Dice coefficient across all test frames (higher is better)
- **Dice≥0.5** — fraction of frames meeting the clinical pass threshold

† YOLO-Crop+UNet crops the YOLO bbox, resizes to 256×256, runs U-Net at higher effective resolution on the glottis region, then projects the mask back to full-frame coordinates.  Requires `unet_glottis_crop.pt` trained with `scripts/train_unet_crop.py` — the two weight files are not interchangeable (full-frame weights on crops collapses to ~0 Dice and vice versa).

YOLO+Motion underperforms because GIRAFE test frames are the first 20 frames per patient, providing insufficient temporal context for the motion tracker to converge.

### Cross-dataset: BAGLS (3 500 test frames, zero-shot transfer)

BAGLS images come in many sizes (256×256 to 512×512); they are letterboxed to 256×256 before inference.  All models were trained on GIRAFE only — no BAGLS training data used.

```bash
python scripts/eval_bagls.py \
    --bagls-dir    BAGLS/test \
    --unet-weights weights/openglottal_unet.pt \
    --crop-weights glottal_detector/unet_glottis_crop.pt \
    --yolo-weights weights/openglottal_yolo.pt \
    --device       mps
```

| Method | Det.Recall | Dice | IoU | Dice≥0.5 |
|--------|-----------|------|-----|----------|
| U-Net only | 1.000 | 0.588 | 0.504 | 67.1% |
| YOLO+UNet | 0.688 | 0.545 | 0.473 | 61.9% |
| **YOLO-Crop+UNet** | **0.688** | **0.609** | **0.533** | **70.3%** |

YOLO-Crop+UNet is the strongest pipeline on the unseen BAGLS data (+2.1pp Dice, +3.2pp Dice≥0.5 over U-Net alone), despite YOLO only detecting on 68.8% of frames (domain shift from GIRAFE).  When YOLO does fire, cropping and re-scaling the region of interest gives U-Net higher effective resolution and cleaner context — benefits that generalise across datasets.

---

## Training

### 1. Build the YOLO dataset

```bash
openglottal build-dataset \
    --images-dir  GIRAFE/Training/imagesTr \
    --labels-dir  GIRAFE/Training/labelsTr \
    --training-json GIRAFE/Training/training.json \
    --output-dir  yolo_data
```

Or via script:

```bash
python scripts/train_yolo.py \
    --images-dir  GIRAFE/Training/imagesTr \
    --labels-dir  GIRAFE/Training/labelsTr \
    --training-json GIRAFE/Training/training.json \
    --epochs 100
```

### 2. Train the U-Net (full-frame mode)

```bash
python scripts/train_unet.py \
    --images-dir  GIRAFE/Training/imagesTr \
    --labels-dir  GIRAFE/Training/labelsTr \
    --training-json GIRAFE/Training/training.json \
    --output glottal_detector/unet_glottis_v2.pt \
    --epochs 50
```

### 3. Train the U-Net (YOLO-crop mode — higher effective resolution)

Uses YOLO to crop each training image to the glottis region, resizes to 256×256, and trains U-Net on these patches.  At inference time the YOLO-Crop+UNet pipeline crops the input, runs U-Net, and projects the mask back to full-frame coordinates.

```bash
python scripts/train_unet_crop.py \
    --images-dir    GIRAFE/Training/imagesTr \
    --labels-dir    GIRAFE/Training/labelsTr \
    --training-json GIRAFE/Training/training.json \
    --yolo-weights  runs/detect/glottal_detector/yolov8n_girafe/weights/best.pt \
    --output        glottal_detector/unet_glottis_crop.pt \
    --crop-size     256 \
    --epochs        50 \
    --device        mps
```

Both training modes use a **50/50 BCE + Dice loss** with cosine annealing, saving the best validation checkpoint automatically.

---

## Repository Structure

```
openglottal/
├── weights/                 # openglottal_yolo.pt, openglottal_unet.pt (bundled)
├── openglottal/
│   ├── models/
│   │   ├── detector.py     # TemporalDetector — YOLOv8 + temporal box locking
│   │   ├── tracker.py      # VocalFoldTracker, YOLOGuidedVFT
│   │   └── unet.py         # UNet, DoubleConv, GlottisDataset (with augmentation)
│   ├── features.py         # extract_features_{detector,yolo_guided_vft,unet}
│   ├── data.py             # mask_to_yolo, build_yolo_dataset
│   ├── utils.py            # I/O helpers, dice/IoU metrics, unet_segment_frame
│   └── cli.py              # `openglottal` command-line interface
├── scripts/
│   ├── train_yolo.py       # standalone YOLO training script
│   ├── train_unet.py       # standalone U-Net training script
│   ├── infer.py            # batch inference: AVI dir or image sequence → _out.avi
│   ├── eval_girafe.py      # per-patient test evaluation vs GIRAFE baselines
│   ├── eval_bagls.py       # cross-dataset evaluation on BAGLS (3 500 frames)
│   ├── analyze_gaw.py      # GAW feature analysis: Healthy vs Pathological (65 patients)
│   ├── download_datasets.py # download GIRAFE and BAGLS from Zenodo
│   ├── make_montage.py     # build frame montage PNGs for paper figures
│   ├── sweep_bagls_conf.py # YOLO confidence threshold sweep on BAGLS
│   └── train_unet_crop.py  # train U-Net on YOLO-cropped patches (higher res)
└── configs/
    └── default.yaml        # all hyperparameters documented in one place
```

---

## Dataset

OpenGlottal is developed and evaluated on **GIRAFE** and **BAGLS**. Download them (optional: use the script) then point the training/eval scripts at the extracted directories.

```bash
# Download GIRAFE and/or BAGLS to the current directory (or use --output-dir)
python scripts/download_datasets.py --girafe --bagls
```

- **GIRAFE** (Zenodo): [zenodo.org/records/13773163](https://zenodo.org/records/13773163) — 760 frames (256×256 px), expert-annotated glottal masks. Dataset paper: [Data in Brief (2025)](https://doi.org/10.1016/j.dib.2024.111376). After unpacking: `GIRAFE/Training/imagesTr/`, `GIRAFE/Training/labelsTr/`, `GIRAFE/Training/training.json`; raw videos: `GIRAFE/Raw_Data/`.
- **BAGLS** (Zenodo): [zenodo.org/record/3381469](https://zenodo.org/record/3381469) — benchmark for automatic glottis segmentation. Dataset paper: [Gómez et al., Scientific Data (2020)](https://doi.org/10.1038/s41597-020-0526-3). Use the test set path as `--bagls-dir` for `eval_bagls.py` and `sweep_bagls_conf.py`.

| Split (GIRAFE) | Frames |
|----------------|--------|
| Train | ~608 |
| Val | ~76 |
| Test | ~76 |

---

## Kinematic Features

The following scalar features are extracted from each glottal area waveform:

| Feature | Description |
|---------|-------------|
| `area_mean` | Mean glottal area (px²) |
| `area_std` | Standard deviation of area |
| `area_range` | Max − min area |
| `open_quotient` | Fraction of cycle with area above 10 % of mean |
| `f0` | Dominant frequency from FFT (cycles/frame; multiply by capture fps for Hz) |
| `periodicity` | Peak autocorrelation at lags 1–50 |
| `cv` | Coefficient of variation (std / mean) |

---

## Glottal Area Waveform Analysis

Beyond frame-level segmentation, the pipeline produces a **Glottal Area Waveform (GAW)** — the per-frame glottal area over time — from which kinematic features can be extracted and used for clinical classification.

```bash
python scripts/analyze_gaw.py \
    --raw-data-dir  GIRAFE/Raw_Data \
    --yolo-weights  runs/detect/glottal_detector/yolov8n_girafe/weights/best.pt \
    --unet-weights  glottal_detector/unet_glottis_v2.pt \
    --device        mps \
    --output-dir    results/gaw
```

This script processes all 65 GIRAFE patients, extracts kinematic features from each area waveform, and compares **Healthy** vs **Pathological** groups using Mann-Whitney U tests.

### Key Findings (65 patients: 15 Healthy, 25 Pathological, 25 Unknown)

Because the cohort has a significant sex imbalance (see note below), results are reported **stratified by sex** rather than pooled.

**Female subgroup (12 H / 11 P):**

| Feature | Healthy (mean±std) | Pathological (mean±std) | p-value |
|---------|-------------------|------------------------|---------|
| area_mean | 125.2 ± 43.1 | 247.8 ± 204.6 | 0.230 |
| area_std | 112.9 ± 32.2 | 118.9 ± 96.0 | 0.406 |
| area_range | 336.7 ± 97.6 | 375.5 ± 272.2 | 0.559 |
| open_quotient | 0.760 ± 0.207 | 0.874 ± 0.131 | 0.192 |
| f0 | 241.7 ± 34.8 Hz | 203.5 ± 73.6 Hz | 0.156 |
| periodicity | 0.955 ± 0.008 | 0.946 ± 0.013 | 0.255 |
| cv | **0.95 ± 0.20** | **0.57 ± 0.29** | **0.006*** |

**Male subgroup (3 H / 14 P):**

| Feature | Healthy (mean±std) | Pathological (mean±std) | p-value |
|---------|-------------------|------------------------|---------|
| area_mean | 192.1 ± 18.3 | 172.7 ± 94.0 | 0.768 |
| area_std | 142.7 ± 35.0 | 92.0 ± 66.9 | 0.197 |
| area_range | 439.7 ± 86.7 | 343.1 ± 212.3 | 0.488 |
| open_quotient | 0.860 ± 0.145 | 0.843 ± 0.186 | 1.000 |
| f0 | 183.3 ± 75.0 Hz | 82.5 ± 79.3 Hz | 0.169 |
| periodicity | 0.962 ± 0.001 | 0.900 ± 0.116 | 0.068 |
| cv | 0.75 ± 0.19 | 0.63 ± 0.40 | 0.509 |

\* p < 0.05 (Mann-Whitney U, two-sided)

> **Sex imbalance note:** The Healthy group is 80% female (12F/3M) while the Pathological group is 56% male (14M/11F; Fisher's exact p=0.025). Because f0 is strongly sex-dependent (males ~100 Hz vs females ~224 Hz), pooling would confound f0. After stratifying, only **cv** (coefficient of variation) reaches significance in the female subgroup (p=0.006). In the male subgroup (n=3 Healthy), cv trends in the same direction (0.75 vs 0.63) but does not reach significance (p=0.509); periodicity approaches significance (p=0.068). The male subgroup is too small for reliable inference.

### Production Robustness

YOLO acts as a **detection gate**: when the endoscope moves away from the glottis (scope insertion, patient coughing, instrument in view), YOLO fires no detection and the area is set to zero — preventing spurious waveform spikes that would corrupt downstream feature extraction.

---

## Citation

If you use OpenGlottal in your research, please cite the underlying work:

```bibtex
@article{patel2013invivo,
  title   = {In vivo measurement of pediatric vocal fold motion using
             structured light laser projection},
  author  = {Patel, Rita R and Donohue, Kevin D and Lau, Daniel and
             Unnikrishnan, Harikrishnan},
  journal = {The Laryngoscope},
  year    = {2013}
}

@article{patel2016effects,
  title   = {Effects of vocal fold nodules on glottal cycle measurements
             derived from high-speed digital imaging},
  author  = {Patel, Rita R and Unnikrishnan, Harikrishnan and Donohue, Kevin D},
  journal = {Journal of Speech, Language, and Hearing Research},
  year    = {2016}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
