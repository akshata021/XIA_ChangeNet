"""PyTorch Dataset for xBD change detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class XBDChangeDataset(Dataset):
    """Dataset returning pre/post disaster tensors and mask."""

    def __init__(
        self,
        root: str | Path,
        pairs_file: str | Path,
        img_size: int = 512,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.pairs_file = Path(pairs_file)
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")
        with self.pairs_file.open("r", encoding="utf-8") as f:
            self.samples: List[dict] = json.load(f)
        if not self.samples:
            raise ValueError(f"Pairs file {self.pairs_file} is empty.")

        base_transforms = [A.Resize(img_size, img_size), A.Normalize()]
        if augment:
            # Stronger augmentation set for training
            aug = [
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.15),
                A.RandomRotate90(p=0.15),
                # Use Affine rather than ShiftScaleRotate to avoid deprecation warnings
                A.Affine(translate_percent=0.04, scale=(0.95, 1.05), rotate=(-10, 10), p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.15),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(p=0.1),
                # Use CoarseDropout with sensible defaults to avoid compatibility warnings
                A.CoarseDropout(p=0.15),
            ]
        else:
            aug = []
        self.transform = A.Compose(aug + base_transforms, additional_targets={"post": "image", "mask": "mask"})
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> np.ndarray:
        path = self.root / rel_path
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))

    def _load_mask(self, rel_path: str) -> np.ndarray:
        path = self.root / rel_path
        with Image.open(path) as img:
            return np.array(img.convert("L"))

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        pre = self._load_image(sample["pre_image"])
        post = self._load_image(sample["post_image"])
        mask = self._load_mask(sample["mask"])

        augmented = self.transform(image=pre, post=post, mask=mask)
        pre_aug = augmented["image"]
        post_aug = augmented["post"]
        mask_aug = augmented["mask"]

        pre_tensor = self.to_tensor(image=pre_aug)["image"]
        post_tensor = self.to_tensor(image=post_aug)["image"]
        mask_tensor = torch.from_numpy(mask_aug).unsqueeze(0).float() / 255.0
        return pre_tensor, post_tensor, mask_tensor


__all__ = ["XBDChangeDataset"]
