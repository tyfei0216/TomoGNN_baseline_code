from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        arr = np.array(Image.open(path), dtype=np.float32)
    elif suffix in {".mrc", ".rec", ".map"}:
        import mrcfile

        with mrcfile.open(path, permissive=True) as m:
            arr = m.data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    return arr.astype(np.float32)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _resize_2d(arr: np.ndarray, target_hw: tuple[int, int], is_mask: bool) -> np.ndarray:
    mode = Image.NEAREST if is_mask else Image.BILINEAR
    pil = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = pil.resize((target_hw[1], target_hw[0]), resample=mode)
    out = np.array(resized, dtype=np.float32)
    return (out > 0.5).astype(np.float32) if is_mask else out


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        augment: bool = False,
    ) -> None:
        super().__init__()
        if images.shape != masks.shape:
            raise ValueError("images and masks must have the same shape")

        self.images = images.astype(np.float32)
        self.masks = (masks > 0.5).astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        y = self.masks[idx]

        if self.augment:
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=0).copy()
                y = np.flip(y, axis=0).copy()
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
            if np.random.rand() < 0.5:
                k = np.random.randint(1, 4)
                x = np.rot90(x, k=k).copy()
                y = np.rot90(y, k=k).copy()

        x_t = torch.from_numpy(x[None, ...])
        y_t = torch.from_numpy(y[None, ...])
        return x_t, y_t


def load_from_metadata(
    metadata_csv: Path,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    class_col: Optional[str] = None,
    filter_class: Optional[int] = None,
    target_shape: Optional[tuple[int, int]] = None,
    return_slice_records: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(metadata_csv)

    if class_col is not None and filter_class is not None:
        df = df[df[class_col] == filter_class].reset_index(drop=True)

    if image_col not in df.columns or mask_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{image_col}' and '{mask_col}'. Found: {list(df.columns)}"
        )

    images = []
    masks = []
    slice_records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        img_path = Path(row[image_col])
        mask_path = Path(row[mask_col])
        if not img_path.exists() or not mask_path.exists():
            continue

        img = _read_array(img_path)
        mask = (_read_array(mask_path) > 0).astype(np.float32)

        if img.shape != mask.shape:
            continue

        # Use all 2D/3D slices that contain positive labels.
        if img.ndim == 2:
            candidate_slices = [(-1, img, mask)] if np.any(mask > 0) else []
        elif img.ndim == 3:
            candidate_slices = []
            for z in range(mask.shape[0]):
                if np.any(mask[z] > 0):
                    candidate_slices.append((z, img[z], mask[z]))
        else:
            continue

        for slice_idx, img_slice, mask_slice in candidate_slices:
            img_slice = _normalize01(img_slice)
            mask_slice = (mask_slice > 0).astype(np.float32)

            if target_shape is not None:
                img_slice = _resize_2d(img_slice, target_shape, is_mask=False)
                mask_slice = _resize_2d(mask_slice, target_shape, is_mask=True)

            images.append(img_slice)
            masks.append(mask_slice)
            slice_records.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "slice_index": int(slice_idx),
                    "positive_pixels": int(mask_slice.sum()),
                }
            )

    if not images:
        raise RuntimeError("No valid image/mask pairs found in metadata CSV.")

    images_np = np.stack(images)
    masks_np = np.stack(masks)

    if return_slice_records:
        return images_np, masks_np, pd.DataFrame(slice_records)
    return images_np, masks_np


def synthetic_blobs(
    n_samples: int = 64,
    image_size: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, image_size, image_size), dtype=np.float32)
    masks = np.zeros_like(images)

    yy, xx = np.mgrid[0:image_size, 0:image_size]

    for i in range(n_samples):
        n_blobs = int(rng.integers(1, 5))
        mask = np.zeros((image_size, image_size), dtype=np.float32)

        for _ in range(n_blobs):
            cy = int(rng.integers(12, image_size - 12))
            cx = int(rng.integers(12, image_size - 12))
            r = int(rng.integers(5, 16))
            blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            mask = np.maximum(mask, blob.astype(np.float32))

        img = mask * rng.uniform(0.6, 1.0) + rng.normal(0.15, 0.1, size=mask.shape)
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        images[i] = img
        masks[i] = mask

    return images, masks
