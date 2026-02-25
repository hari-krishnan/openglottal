"""YOLO dataset construction from GIRAFE-format segmentation masks.

Also: Kaggle dataset path resolution and HDF5-backed Glottis dataset for efficient I/O.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from openglottal.kaggle_paths import get_kaggle_bagls_path
from openglottal.utils import letterbox_with_info, letterbox_apply_geometry

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

IMG_W = IMG_H = 256   # all GIRAFE frames are 256×256
DILATE = 10           # pixels added on each side of the tight mask bbox


def mask_to_yolo(
    mask_path: str | Path,
    dilate: int = DILATE,
    img_wh: tuple[int, int] | None = None,
) -> str:
    """
    Convert a binary segmentation mask to a YOLO bbox label string.

    Parameters
    ----------
    mask_path:
        Path to a grayscale PNG mask where glottis pixels are > 0.
    dilate:
        Number of pixels to expand the tight bounding box on each side.
    img_wh:
        (width, height) for normalisation. If None, use 256×256 (GIRAFE).
        For variable-size (e.g. BAGLS), pass (mask.shape[1], mask.shape[0]).

    Returns
    -------
    A YOLO-format label string ``"0 cx cy w h"`` (normalised), or ``""``
    if the mask is empty.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return ""
    H, W = mask.shape[:2]
    if img_wh is not None:
        W, H = img_wh
    else:
        W, H = IMG_W, IMG_H
    ys, xs = np.where(mask > 0)
    x1 = max(0, xs.min() - dilate)
    x2 = min(W, xs.max() + dilate)
    y1 = max(0, ys.min() - dilate)
    y2 = min(H, ys.max() + dilate)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    training_json: str | Path,
    output_dir: str | Path,
    dilate: int = DILATE,
    force: bool = False,
    mask_suffix: str = "",
) -> Path:
    """
    Build a YOLO-format dataset from GIRAFE images and masks.

    Directory layout created under ``output_dir``::

        output_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Parameters
    ----------
    images_dir:
        Directory containing PNG frames (e.g. ``GIRAFE/Training/imagesTr``).
    labels_dir:
        Directory containing PNG masks (e.g. ``GIRAFE/Training/labelsTr``).
    training_json:
        Path to the split JSON with keys ``training``, ``Val``, and optionally ``test``.
    mask_suffix:
        If set (e.g. ``"_seg"`` for BAGLS), mask path is ``stem + mask_suffix + ".png"``.
    output_dir:
        Root directory for the generated YOLO dataset.
    dilate:
        Pixel dilation applied to each mask bbox.
    force:
        Rebuild even if the dataset already exists.

    Returns
    -------
    Path to the generated ``dataset.yaml`` file.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    splits = json.load(open(training_json))
    split_map = {
        "train": splits["training"],
        "val": splits["Val"],
        "test": splits.get("test", []),
    }

    def _complete() -> bool:
        for split in split_map:
            if not (output_dir / "images" / split).exists():
                return False
            if not (output_dir / "labels" / split).exists():
                return False
        return True

    if _complete() and not force:
        print("Dataset already exists — skipping build. Pass force=True to rebuild.")
    else:
        for split, fnames in split_map.items():
            img_out = output_dir / "images" / split
            lbl_out = output_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for fname in fnames:
                shutil.copy(images_dir / fname, img_out / fname)
                stem = Path(fname).stem
                mask_path = labels_dir / f"{stem}{mask_suffix}.png" if mask_suffix else labels_dir / fname
                # Variable-size (e.g. BAGLS): use mask dimensions for YOLO norm
                img_wh = None
                if mask_suffix and mask_path.exists():
                    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        img_wh = (m.shape[1], m.shape[0])
                label = mask_to_yolo(mask_path, dilate=dilate, img_wh=img_wh)
                (lbl_out / f"{stem}.txt").write_text(label)
        print(f"Dataset built at {output_dir}")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: 1\n"
        f"names: ['glottis']\n"
    )
    return yaml_path


# ── Kaggle dataset paths ─────────────────────────────────────────────────────

def resolve_kaggle_data_paths(
    dataset: str,
    split: str,
) -> tuple[Path, Path] | None:
    """
    Resolve (images_dir, labels_dir) when using a Kaggle dataset.

    Parameters
    ----------
    dataset : str
        Currently only ``"bagls"`` is supported.
    split : str
        ``"training"`` or ``"test"``.

    Returns
    -------
    (img_dir, lbl_dir) if on Kaggle and the dataset is available, else None.
    For BAGLS, images and masks are in the same directory (N.png, N_seg.png).
    """
    if dataset.lower() != "bagls":
        return None
    path = get_kaggle_bagls_path(split)
    if path is None:
        return None
    return path, path


# ── HDF5 cache for letterboxed images (efficient I/O) ────────────────────────

def build_glottis_hdf5(
    fnames: list[str],
    img_dir: str | Path,
    lbl_dir: str | Path,
    output_path: str | Path,
    label_suffix: str = "",
    size: int = 256,
) -> Path:
    """
    Write letterboxed images and masks to an HDF5 file for fast loading.

    Parameters
    ----------
    fnames : list[str]
        Image filenames (e.g. from training.json).
    img_dir, lbl_dir : path
        Directories containing PNGs.
    output_path : path
        Output .h5 path (e.g. ``outputs/bagls_train.h5``).
    label_suffix : str
        Mask filename is ``stem + label_suffix + ".png"`` (e.g. ``"_seg"`` for BAGLS).
    size : int
        Letterbox target size (default 256).

    Returns
    -------
    Path to the written HDF5 file.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 cache. Install with: pip install h5py")

    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(fnames)
    with h5py.File(output_path, "w") as f:
        images = f.create_dataset("images", (n, size, size), dtype="u1", chunks=(1, size, size))
        masks = f.create_dataset("masks", (n, size, size), dtype="u1", chunks=(1, size, size))
        fnames_ds = f.create_dataset("fnames", (n,), dtype=h5py.special_dtype(vlen=str))
        for i, fname in enumerate(fnames):
            stem = Path(fname).stem
            lbl_name = f"{stem}{label_suffix}.png" if label_suffix else fname
            img = cv2.imread(str(img_dir / fname), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(lbl_dir / lbl_name), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                raise FileNotFoundError(f"Missing image or mask: {fname} / {lbl_name}")
            h, w = img.shape[:2]
            if (h, w) != (size, size):
                img_boxed, pad_t, pad_l, content_h, content_w = letterbox_with_info(img, size, value=0)
                msk_boxed = letterbox_apply_geometry(
                    msk, size, pad_t, pad_l, content_h, content_w,
                    value=0, interp=cv2.INTER_NEAREST,
                )
                img, msk = img_boxed, msk_boxed
            images[i] = img
            masks[i] = (msk > 0).astype("uint8")
            fnames_ds[i] = fname
    return output_path


class GlottisDatasetHDF5(Dataset):
    """
    Dataset that reads letterboxed images and masks from an HDF5 file (efficient I/O).

    Use :func:`build_glottis_hdf5` to create the file from raw PNGs. Optionally
    apply the same augmentations as :class:`openglottal.models.GlottisDataset`
    at load time.
    """

    SIZE = 256

    def __init__(self, h5_path: str | Path, augment: bool = False) -> None:
        if h5py is None:
            raise ImportError("h5py is required. Install with: pip install h5py")
        self._h5_path = Path(h5_path)
        self.augment = augment
        with h5py.File(self._h5_path, "r") as f:
            self._n = f["images"].shape[0]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple:
        import random

        import torch
        import torchvision.transforms.functional as TF

        with h5py.File(self._h5_path, "r") as f:
            img = f["images"][idx]
            msk = f["masks"][idx]
        img = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)  # (1,H,W)
        msk = torch.from_numpy(msk.astype("float32")).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if random.random() > 0.5:
                img, msk = TF.vflip(img), TF.vflip(msk)
            angle = random.uniform(-30, 30)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            msk = TF.rotate(msk, angle, interpolation=TF.InterpolationMode.NEAREST)
            if random.random() > 0.5:
                scale = random.uniform(0.85, 1.15)
                new_size = int(256 * scale)
                img = TF.resize(img, [new_size, new_size], interpolation=TF.InterpolationMode.BILINEAR)
                msk = TF.resize(msk, [new_size, new_size], interpolation=TF.InterpolationMode.NEAREST)
                if new_size > 256:
                    off = (new_size - 256) // 2
                    img = TF.crop(img, off, off, 256, 256)
                    msk = TF.crop(msk, off, off, 256, 256)
                else:
                    pad = 256 - new_size
                    pl, pr = pad // 2, pad - pad // 2
                    img = TF.pad(img, [pl, pl, pr, pr])
                    msk = TF.pad(msk, [pl, pl, pr, pr])
            if random.random() > 0.5:
                sigma = random.uniform(0.01, 0.05)
                img = torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)
            if random.random() > 0.5:
                ks = random.choice([3, 5])
                sigma = random.uniform(0.5, 1.5)
                img = TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)
            if random.random() > 0.5:
                img = torch.clamp(img * random.uniform(0.7, 1.3), 0.0, 1.0)
            if random.random() > 0.5:
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        return img, msk
