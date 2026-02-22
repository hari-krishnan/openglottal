"""Lightweight U-Net for binary glottal segmentation + GIRAFE dataset loader."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for binary glottal segmentation (1-channel input/output).

    Parameters
    ----------
    in_ch:
        Number of input channels (1 for grayscale).
    out_ch:
        Number of output channels (1 for binary segmentation).
    features:
        Channel sizes at each encoder stage.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(ch, ch * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], out_ch, 1)  # raw logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i // 2 + 1)]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, s.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        return self.head(x)


class GlottisDataset(Dataset):
    """
    Grayscale 256×256 GIRAFE frames + binary GT masks.

    Parameters
    ----------
    fnames:
        List of filename stems (e.g. from ``training.json``).
    img_dir:
        Directory containing PNG frames.
    lbl_dir:
        Directory containing PNG masks.
    augment:
        Whether to apply random augmentation at load time.
    """

    def __init__(
        self,
        fnames: list[str],
        img_dir: str | Path,
        lbl_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.fnames = fnames
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.fnames[idx]
        img = cv2.imread(str(self.img_dir / fname), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(self.lbl_dir / fname), cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)  # (1,H,W)
        msk = torch.from_numpy((msk > 0).astype("float32")).unsqueeze(0)    # (1,H,W)

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
                img = TF.resize(img, [new_size, new_size],
                                interpolation=TF.InterpolationMode.BILINEAR)
                msk = TF.resize(msk, [new_size, new_size],
                                interpolation=TF.InterpolationMode.NEAREST)
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
