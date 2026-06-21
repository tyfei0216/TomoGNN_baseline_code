from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def build_sam2_predictor(
    sam2_repo: str | Path,
    model_cfg: str | Path,
    checkpoint: str | Path,
    device: torch.device,
):
    import sys

    sam2_repo = str(Path(sam2_repo))
    if sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_cfg = Path(model_cfg)
    checkpoint = Path(checkpoint)

    if model_cfg.exists():
        # Allow local YAML config files outside Hydra's package search path.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(model_cfg.parent), version_base=None):
            cfg = compose(config_name=model_cfg.stem)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model = model.to(device)
        model.eval()
    else:
        model = build_sam2(str(model_cfg), str(checkpoint), device=str(device))

    predictor = SAM2ImagePredictor(model)
    return predictor


def freeze_for_finetune(predictor) -> None:
    predictor.model.image_encoder.train(False)
    for p in predictor.model.image_encoder.parameters():
        p.requires_grad = False

    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.sam_mask_decoder.train(True)


def tensor_batch_to_np_rgb(images_t: torch.Tensor) -> list[np.ndarray]:
    images_np = []
    for b in range(images_t.size(0)):
        img = images_t[b].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        images_np.append(img)
    return images_np


@torch.no_grad()
def set_batch_embeddings(predictor, images_t: torch.Tensor) -> None:
    predictor.set_image_batch(tensor_batch_to_np_rgb(images_t))


def forward_batch_logits(predictor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits_list = []
    scores_list = []

    for b in range(batch_size):
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=None, boxes=None, masks=None
        )

        high_res_features = [
            feat_level[b].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
        ]
        image_embeddings = predictor._features["image_embed"][b].unsqueeze(0)

        low_res_masks, pred_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        pred_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[b])
        logits_list.append(pred_masks[:, 0])
        scores_list.append(pred_scores[:, 0])

    logits = torch.cat(logits_list, dim=0)
    scores = torch.cat(scores_list, dim=0)
    return logits, scores


def _batch_iou_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    inter = (masks * preds).sum(dim=(1, 2))
    union = masks.sum(dim=(1, 2)) + preds.sum(dim=(1, 2)) - inter
    return (inter + 1e-6) / (union + 1e-6)


def fit_sam2_one_fold(
    predictor,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: str | Path,
    max_train_steps: int | None = None,
    max_val_batches: int | None = None,
) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freeze_for_finetune(predictor)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, predictor.model.parameters()),
        lr=lr,
        weight_decay=4e-5,
    )
    bce_logits = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        predictor.model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for step, (images, masks) in enumerate(pbar):
            if max_train_steps is not None and step >= max_train_steps:
                break

            images = images.to(device)
            masks = masks.to(device)

            set_batch_embeddings(predictor, images)

            amp_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
            with amp_ctx:
                logits, pred_scores = forward_batch_logits(predictor, batch_size=images.size(0))
                seg_loss = bce_logits(logits, masks)
                with torch.no_grad():
                    iou = _batch_iou_from_logits(logits, masks)
                score_loss = torch.abs(pred_scores - iou).mean()
                loss = seg_loss + 0.05 * score_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))

        predictor.model.eval()
        val_ious = []
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Eval", leave=False)):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break

                images = images.to(device)
                masks = masks.to(device)

                set_batch_embeddings(predictor, images)
                logits, _ = forward_batch_logits(predictor, batch_size=images.size(0))
                iou = _batch_iou_from_logits(logits, masks)
                val_ious.extend(iou.detach().cpu().tolist())

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else float("nan")),
            "val_iou": float(np.mean(val_ious) if val_ious else float("nan")),
        }
        history.append(row)
        print(f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | val_iou={row['val_iou']:.4f}")

        ckpt_path = out_dir / f"sam2_small_onefold_epoch{epoch}.pt"
        torch.save(predictor.model.state_dict(), ckpt_path)

    return history
