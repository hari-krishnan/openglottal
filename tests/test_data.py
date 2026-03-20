"""Tests for HDF5-backed Glottis dataset helpers in openglottal.data."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def test_build_glottis_hdf5_and_load(tmp_path: Path) -> None:
    """Build HDF5 from small fake PNGs and load via GlottisDatasetHDF5."""
    from openglottal.data import GlottisDatasetHDF5, build_glottis_hdf5

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    size = 64  # small for test
    fnames = ["a.png", "b.png"]
    for i, fname in enumerate(fnames):
        stem = Path(fname).stem
        img = np.full((size, size), 100 + i * 50, dtype=np.uint8)
        msk = np.zeros((size, size), dtype=np.uint8)
        msk[16:48, 16:48] = 255
        cv2.imwrite(str(img_dir / fname), img)
        cv2.imwrite(str(lbl_dir / f"{stem}_seg.png"), msk)

    out_h5 = tmp_path / "cache.h5"
    build_glottis_hdf5(
        fnames,
        img_dir,
        lbl_dir,
        out_h5,
        label_suffix="_seg",
        size=256,
    )
    assert out_h5.exists()

    ds = GlottisDatasetHDF5(out_h5, augment=False)
    assert len(ds) == 2
    img, msk = ds[0]
    assert img.shape == (1, 256, 256)
    assert msk.shape == (1, 256, 256)
    assert 0 <= img.min() <= img.max() <= 1.0
    assert msk.min() in (0.0, 1.0) and msk.max() in (0.0, 1.0)

    img1, msk1 = ds[1]
    assert img1.shape == (1, 256, 256)
    assert msk1.shape == (1, 256, 256)


def test_glottis_dataset_hdf5_augment_true(tmp_path: Path) -> None:
    """GlottisDatasetHDF5 with augment=True returns valid tensors."""
    from openglottal.data import GlottisDatasetHDF5, build_glottis_hdf5

    img_dir = tmp_path / "im"
    lbl_dir = tmp_path / "lb"
    img_dir.mkdir()
    lbl_dir.mkdir()
    cv2.imwrite(str(img_dir / "x.png"), np.zeros((64, 64), dtype=np.uint8))
    cv2.imwrite(str(lbl_dir / "x_seg.png"), np.ones((64, 64), dtype=np.uint8) * 255)

    h5_path = tmp_path / "tiny.h5"
    build_glottis_hdf5(["x.png"], img_dir, lbl_dir, h5_path, label_suffix="_seg", size=256)

    ds = GlottisDatasetHDF5(h5_path, augment=True)
    img, msk = ds[0]
    assert img.shape == (1, 256, 256)
    assert msk.shape == (1, 256, 256)
    assert float(img.min()) >= 0 and float(img.max()) <= 1.5  # contrast can push slightly >1
