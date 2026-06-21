from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mrcfile
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def _normalize_to_uint8(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(img, pmin))
    hi = float(np.percentile(img, pmax))
    img = np.clip(img, lo, hi)
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - lo) / (hi - lo)
    return (img * 255.0).astype(np.uint8)


def _read_volume(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as m:
        arr = np.asarray(m.data)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr.shape} for {path}")
    return arr


def build_manifest_from_metadata(
    metadata_csv: Path | str,
    out_root: Path | str,
    image_col: str = "data",
    mask_col: str = "labels",
    max_volumes: int | None = None,
    max_slices_per_volume: int | None = None,
) -> Path:
    """Convert metadata-based tomograms into 2D PNG slices for SAM2 finetuning."""
    metadata_csv = Path(metadata_csv)
    out_root = Path(out_root)
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    if image_col not in df.columns or mask_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns {image_col!r} and {mask_col!r}. Found: {list(df.columns)}"
        )

    if max_volumes is not None:
        df = df.iloc[: max(0, int(max_volumes))].copy()

    manifest_rows: list[dict[str, object]] = []

    for row_idx, row in df.reset_index(drop=True).iterrows():
        data_path = Path(row[image_col])
        label_path = Path(row[mask_col])
        if not data_path.exists() or not label_path.exists():
            continue

        volume = _read_volume(data_path).astype(np.float32)
        labels = _read_volume(label_path).astype(np.uint8)
        if volume.shape != labels.shape:
            continue

        kept = 0
        sample_id = data_path.stem

        for z in range(volume.shape[0]):
            mask_slice = labels[z]
            if np.count_nonzero(mask_slice) == 0:
                continue

            if max_slices_per_volume is not None and kept >= int(max_slices_per_volume):
                break

            img_slice = _normalize_to_uint8(volume[z])
            img_name = f"{sample_id}_z{z:04d}.png"
            mask_name = f"{sample_id}_z{z:04d}_mask.png"
            img_path = images_dir / img_name
            mask_path = masks_dir / mask_name

            Image.fromarray(img_slice).save(img_path)
            Image.fromarray(mask_slice).save(mask_path)

            manifest_rows.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "id": sample_id,
                    "z": int(z),
                    "row_index": int(row_idx),
                }
            )
            kept += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


class SAM2SliceDataset(Dataset):
    def __init__(
        self,
        rows: Iterable[dict[str, object]],
        class_id: int = 1,
        target_size: tuple[int, int] = (1024, 1024),
        augment: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.class_id = int(class_id)
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> np.ndarray:
        img_pil = Image.open(path).convert("L")
        if img_pil.size != self.target_size:
            img_pil = img_pil.resize(self.target_size, resample=Image.BILINEAR)
        img_pil = ImageOps.equalize(img_pil)
        img = np.asarray(img_pil, dtype=np.float32) / 255.0
        return np.stack([img, img, img], axis=0)

    def _load_mask(self, path: str) -> np.ndarray:
        mask_pil = Image.open(path)
        if mask_pil.size != self.target_size:
            mask_pil = mask_pil.resize(self.target_size, resample=Image.NEAREST)
        mask = np.asarray(mask_pil, dtype=np.int64)
        mask = (mask == self.class_id).astype(np.float32)
        return mask

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = self._load_image(str(row["image_path"]))
        mask = self._load_mask(str(row["mask_path"]))

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                k = int(np.random.randint(1, 4))
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


def one_fold_split(manifest_df: pd.DataFrame, train_tomos: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    tomos = sorted(manifest_df["id"].unique().tolist())
    if len(tomos) < 2:
        raise ValueError("Need at least 2 tomograms for one-fold split.")

    train_ids = tomos[: min(train_tomos, len(tomos) - 1)]
    val_id = tomos[-1]

    train_df = manifest_df[manifest_df["id"].isin(train_ids)].reset_index(drop=True)
    val_df = manifest_df[manifest_df["id"] == val_id].reset_index(drop=True)
    return train_df, val_df
