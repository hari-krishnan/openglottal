#!/usr/bin/env bash
# Run BAGLS zero-shot evaluation at YOLO confidence threshold tau=0.02
# (full 3500 test frames). Output matches the zero-shot table in the paper.

set -e
cd "$(dirname "$0")/.."

BAGLS_DIR="${BAGLS_DIR:-BAGLS/test}"
UNET_WEIGHTS="${UNET_WEIGHTS:-outputs/openglottal_unet.pt}"
CROP_WEIGHTS="${CROP_WEIGHTS:-outputs/openglottal_unet_crop.pt}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-outputs/openglottal_yolo.pt}"
DEVICE="${DEVICE:-mps}"

if [[ ! -d "$BAGLS_DIR" ]]; then
  echo "BAGLS test dir not found: $BAGLS_DIR"
  echo "Set BAGLS_DIR or download BAGLS and point to test/ (e.g. BAGLS/test)"
  exit 1
fi

python3 scripts/eval_bagls.py \
  --bagls-dir   "$BAGLS_DIR" \
  --unet-weights "$UNET_WEIGHTS" \
  --crop-weights "$CROP_WEIGHTS" \
  --yolo-weights "$YOLO_WEIGHTS" \
  --device      "$DEVICE" \
  --conf        0.02 \
  --crop-pad    0
