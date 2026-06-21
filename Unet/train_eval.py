from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    iou: float


def binary_iou(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    ious = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        ious.extend(binary_iou(logits, masks).detach().cpu().numpy().tolist())

    return EpochMetrics(loss=float(np.mean(losses)), iou=float(np.mean(ious)))


def fit_unet(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    ckpt_path: Path | str = Path("best_unet_baseline.pth"),
) -> list[dict[str, Any]]:
    ckpt_path = Path(ckpt_path)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, Any]] = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_iou": train_metrics.iou,
            "val_loss": val_metrics.loss,
            "val_iou": val_metrics.iou,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics.loss:.4f}, train_iou={train_metrics.iou:.4f} | "
            f"val_loss={val_metrics.loss:.4f}, val_iou={val_metrics.iou:.4f}"
        )

        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved new best checkpoint: {ckpt_path}")

    return history


def evaluate_unet(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ckpt_path: Path | str,
) -> dict[str, float]:
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        metrics = run_epoch(model, loader, criterion, device, optimizer=None)

    return {"loss": metrics.loss, "mean_iou": metrics.iou}
