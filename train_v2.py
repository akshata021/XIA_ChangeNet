"""Training script (v2) for XAl-ChangeNet with validation and TensorBoard."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import yaml
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.siamese_unet import SiameseResNet18UNet
from utils.data_loader import XBDChangeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XAl-ChangeNet (v2)")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--val_pairs", type=Path, default=None, help="Optional validation pairs manifest (JSON)")
    parser.add_argument("--logdir", type=Path, default=Path("runs"), help="TensorBoard runs dir")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device override: auto|cpu|cuda")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and dataloaders and exit without training (sanity check)")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_pairs_file(root: Path) -> Path:
    matches = sorted(root.glob("pairs_*.json"))
    if not matches:
        raise FileNotFoundError(f"No pairs manifest found in {root}")
    return matches[0]


def combine_pairs(root: Path, out_name: str = "pairs_all.json", exclude: list | None = None) -> Path:
    """Combine all pairs_*.json into a single pairs_all.json, optionally excluding some manifest paths.

    exclude may be a list of Path or str pointing to `pairs_<event>.json` to skip.
    """
    exclude = exclude or []
    exclude = [str(p) for p in exclude]
    # Exclude pairs_all.json to avoid including it in the combination (would cause duplicates)
    exclude.append(str(root / out_name))
    matches = [p for p in sorted(root.glob("pairs_*.json")) if str(p) not in exclude]
    if not matches:
        raise FileNotFoundError(f"No pairs manifest found in {root}")
    combined = []
    for p in matches:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        combined.extend(data)
    out_path = root / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f)
    return out_path


def build_dataloader(cfg: Dict[str, Any], pairs_file: Path | None = None, augment: bool = True, shuffle: bool = True) -> DataLoader:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    # If pairs_file is "all" we combine all pairs under the root
    # Accept optional `pairs_file` Path or handle `pairs_file: 'all'` by combining
    if pairs_file:
        pairs_path = Path(pairs_file)
    else:
        pairs_cfg = data_cfg.get("pairs_file")
        if isinstance(pairs_cfg, str) and pairs_cfg.lower() == "all":
            # If a validation split was specified, exclude it from the training manifest
            val_pairs_cfg = cfg.get("data", {}).get("val_pairs")
            excludes = []
            if val_pairs_cfg:
                # allow single val path or list
                if isinstance(val_pairs_cfg, (list, tuple)):
                    excludes = [str(Path(p)) for p in val_pairs_cfg]
                else:
                    excludes = [str(Path(val_pairs_cfg))]
            pairs_path = combine_pairs(root, exclude=excludes)
        else:
            pairs_path = Path(pairs_cfg) if pairs_cfg else get_pairs_file(root)
    dataset = XBDChangeDataset(root=root, pairs_file=pairs_path, img_size=data_cfg.get("img_size", 512), augment=augment)
    batch_size = data_cfg.get("batch_size", 2) if augment else data_cfg.get("val_batch_size", 1)
    pin_memory = torch.cuda.is_available()
    cfg_workers = data_cfg.get("num_workers", 2)
    # On Windows, avoid spawn hang issues by forcing 0 workers unless specifically set
    import sys
    if sys.platform.startswith("win") and cfg_workers > 0:
        print("Info: Windows detected - setting DataLoader num_workers=0 to avoid spawn-related hangs", flush=True)
        cfg_workers = 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg_workers,
        pin_memory=pin_memory,
        drop_last=augment,
    )


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num + eps) / den


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum().item()
    union = pred_bin.sum().item() + target.sum().item() - inter
    return inter / (union + 1e-6)


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum().item()
    total = pred_bin.sum().item() + target.sum().item()
    return (2 * inter) / (total + 1e-6)


def compute_pos_weight_from_dataset(dataset: XBDChangeDataset) -> float:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    pos = 0
    neg = 0
    for _, _, mask in loader:
        m = mask.squeeze().numpy()
        pos += int(m.sum())
        neg += int(m.size - m.sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, 1, H, W), targets: (N, 1, H, W)
        # Force float32 for numerically stable computations (prevents underflow in float16)
        logits_f = logits.float()
        targets_f = targets.float()
        probs = torch.sigmoid(logits_f)
        pt = torch.where(targets_f == 1, probs, 1 - probs)
        w = torch.where(targets == 1, self.alpha, (1 - self.alpha))
        # Clamp pt to avoid log(0) under low-precision arithmetic
        pt = pt.clamp(min=1e-6)
        loss = -w * ((1 - pt) ** self.gamma) * torch.log(pt)
        if self.reduction == "mean":
            return loss.mean().to(logits.dtype)
        elif self.reduction == "sum":
            return loss.sum().to(logits.dtype)
        else:
            return loss.to(logits.dtype)


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer: optim.Optimizer, scaler: amp.GradScaler, metadata: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def train():
    args = parse_args()
    cfg = load_config(args.config)

    # Device, allow CLI override
    if args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device but not available")
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print device info for verification
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    model = SiameseResNet18UNet().to(device)

    train_loader = build_dataloader(cfg, augment=True)
    val_loader = None
    val_pairs = args.val_pairs or cfg["data"].get("val_pairs")
    if val_pairs:
        val_ds = XBDChangeDataset(root=Path(cfg["data"]["root"]), pairs_file=Path(val_pairs), img_size=cfg["data"].get("img_size", 512), augment=False)
        val_loader = DataLoader(val_ds, batch_size=cfg["data"].get("val_batch_size", 1), shuffle=False, num_workers=cfg["data"].get("num_workers", 2), pin_memory=torch.cuda.is_available())

    if args.dry_run:
        print("Dry-run mode: dataset and dataloaders built successfully.")
        print(f"  Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Val samples: {len(val_loader.dataset)}")
        print("Exiting without training.")
        return

    train_cfg = cfg["training"]
    # Seed everything for reproducibility - after `train_cfg` is available
    seed = train_cfg.get("seed", 42)
    import random
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-4))

    pos_w = train_cfg.get("pos_weight", None)
    if pos_w is None:
        try:
            pos_w = compute_pos_weight_from_dataset(XBDChangeDataset(root=Path(cfg["data"]["root"]), pairs_file=Path(cfg["data"].get("pairs_file") or get_pairs_file(Path(cfg["data"]["root"]))), img_size=cfg["data"].get("img_size", 512), augment=False))
        except Exception:
            pos_w = 1.0
    # clamp pos_weight to prevent extremely large BCE gradients (stable by default)
    max_pos_w = train_cfg.get("max_pos_weight", 50.0)
    if pos_w is None:
        pos_w = 1.0
    pos_w = float(min(pos_w, max_pos_w))

    # select loss
    loss_type = train_cfg.get("loss_type", "bce+dice")
    use_focal = 'focal' in loss_type
    # Combined loss weights: allow config to specify weights for bce, dice, focal
    loss_weights = train_cfg.get('loss_weights', {'bce': 1.0, 'dice': 1.0, 'focal': 1.0})
    # Always create BCE and Focal loss instances; we'll select via weights
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    focal_loss_fn = FocalLoss(alpha=train_cfg.get("focal_alpha", 0.25), gamma=train_cfg.get("focal_gamma", 2.0))
    # Use new torch.amp API (device_type inferred from device) to avoid deprecation warnings
    use_amp = train_cfg.get("use_amp", True)
    scaler = amp.GradScaler() if use_amp else None

    scheduler = None
    if train_cfg.get("scheduler", False):
        stype = train_cfg.get("scheduler_type", "cosine")
        if stype == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.get("epochs", 20))
        elif stype == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.get("scheduler_step_size", 5), gamma=train_cfg.get("scheduler_gamma", 0.1))
        else:
            # Fallback to cosine annealing
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.get("epochs", 20))

    accum_steps = train_cfg.get("accum_steps", 1)
    debug_anomaly = train_cfg.get("debug_anomaly", False)

    # keep a stored config metadata snapshot for checkpointing
    saved_metadata = {
        "seed": int(seed) if 'seed' in locals() else None,
        "cfg": {},
    }
    # Save a selected subset of config to metadata to avoid extremely large files
    saved_metadata["cfg"]["data"] = {k: v for k, v in cfg.get("data", {}).items() if k in ["root", "img_size", "pairs_file", "val_pairs"]}
    saved_metadata["cfg"]["training"] = {k: v for k, v in cfg.get("training", {}).items() if k in ["lr", "epochs", "loss_type", "loss_weights"]}

    writer = SummaryWriter(log_dir=str(args.logdir))
    # Log seed and a small config snapshot to TensorBoard for reproducibility
    try:
        writer.add_text("training/cfg", json.dumps(saved_metadata.get("cfg", {})))
        writer.add_text("training/seed", str(saved_metadata.get("seed", "")))
    except Exception:
        pass

    best_val_iou = -1.0

    for epoch in range(1, train_cfg.get("epochs", 20) + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            did_backward = False
            pre, post, mask = [x.to(device) for x in batch]
            # Autocast uses device_type based on the `device` selected by PyTorch
            if debug_anomaly:
                print('Anomaly Detection: wrapping forward and backward')
                with torch.autograd.detect_anomaly():
                    if use_amp:
                        with amp.autocast(device_type=device.type):
                            logits = model(pre, post)
                            bce_term = bce_loss_fn(logits, mask)
                            focal_term = focal_loss_fn(logits, mask) if use_focal else torch.tensor(0.0, device=device)
                            dice_term = dice_loss(logits, mask).mean()
                            loss = (
                                loss_weights.get('bce', 1.0) * bce_term
                                + loss_weights.get('focal', 1.0) * focal_term
                                + loss_weights.get('dice', 1.0) * dice_term
                            )
                            loss = loss / accum_steps
                    else:
                        logits = model(pre, post)
                        bce_term = bce_loss_fn(logits, mask)
                        focal_term = focal_loss_fn(logits, mask) if use_focal else torch.tensor(0.0, device=device)
                        dice_term = dice_loss(logits, mask).mean()
                        loss = (
                                loss_weights.get('bce', 1.0) * bce_term
                                + loss_weights.get('focal', 1.0) * focal_term
                                + loss_weights.get('dice', 1.0) * dice_term
                        )
                        loss = loss / accum_steps
            else:
                if use_amp:
                    with amp.autocast(device_type=device.type):
                        logits = model(pre, post)
                        bce_term = bce_loss_fn(logits, mask)
                        focal_term = focal_loss_fn(logits, mask) if use_focal else torch.tensor(0.0, device=device)
                        dice_term = dice_loss(logits, mask).mean()
                        loss = (
                            loss_weights.get('bce', 1.0) * bce_term
                            + loss_weights.get('focal', 1.0) * focal_term
                            + loss_weights.get('dice', 1.0) * dice_term
                        )
                        loss = loss / accum_steps
                else:
                    logits = model(pre, post)
                    bce_term = bce_loss_fn(logits, mask)
                    focal_term = focal_loss_fn(logits, mask) if use_focal else torch.tensor(0.0, device=device)
                    dice_term = dice_loss(logits, mask).mean()
                    loss = (
                        loss_weights.get('bce', 1.0) * bce_term
                        + loss_weights.get('focal', 1.0) * focal_term
                        + loss_weights.get('dice', 1.0) * dice_term
                    )
                    loss = loss / accum_steps
                # debug/guard: if loss is NaN or Inf skip this batch and report
                if not torch.isfinite(loss):
                    print(f"Skipping non-finite loss at epoch {epoch} step {step}:", loss)
                    # Print debug terms
                    try:
                        with torch.no_grad():
                            logits_dbg = logits.detach()
                            print("  logits min/max/mean:", float(logits_dbg.min()), float(logits_dbg.max()), float(logits_dbg.mean()))
                            print("  mask sum:", int(mask.sum().item()))
                    except Exception:
                        pass
                    continue
                try:
                    if debug_anomaly:
                        with torch.autograd.detect_anomaly():
                            if use_amp:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                    else:
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    did_backward = True
                except RuntimeError as e:
                    # Print full stack/frame of anomaly and rethrow for debugging or skip
                    print('Anomaly detected during backward:', e)
                    if debug_anomaly:
                        raise
                    else:
                        did_backward = False
                        continue

            if step % accum_steps == 0:
                # gradient clipping to stabilize training
                max_norm = train_cfg.get("max_grad_norm", 1.0)
                if did_backward:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    # Timing the optimizer step to help debug slow hangs
                    t_step_start = time.time()
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    t_step_end = time.time()
                    if t_step_end - t_step_start > 1.0:
                        print(f"Warning: optimizer step took {t_step_end - t_step_start:.2f}s (epoch {epoch} step {step})", flush=True)
                else:
                    # nothing to step if we didn't call backward this accumulation group
                    pass
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        val_iou = None
        val_dice = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            ious, dices, accs = [], [], []
            val_loss = 0.0
            with torch.no_grad():
                all_probs = []
                all_masks = []
                for vbatch in val_loader:
                    pre_v, post_v, mask_v = [x.to(device) for x in vbatch]
                    logits_v = model(pre_v, post_v)
                    probs_v = torch.sigmoid(logits_v)
                    bce_term_v = bce_loss_fn(logits_v, mask_v)
                    focal_term_v = focal_loss_fn(logits_v, mask_v) if use_focal else torch.tensor(0.0, device=device)
                    dice_term_v = dice_loss(logits_v, mask_v).mean()
                    loss_v = (
                        loss_weights.get('bce', 1.0) * bce_term_v
                        + loss_weights.get('focal', 1.0) * focal_term_v
                        + loss_weights.get('dice', 1.0) * dice_term_v
                    )
                    val_loss += loss_v.item()
                    iou_sample = compute_iou((probs_v > 0.5).float(), mask_v)
                    dice_sample = compute_dice((probs_v > 0.5).float(), mask_v)
                    acc_sample = float(((probs_v > 0.5).float() == mask_v).float().mean().item())
                    ious.append(iou_sample)
                    dices.append(dice_sample)
                    accs.append(acc_sample)
                    # collect for threshold optimization
                    all_probs.append(probs_v.squeeze().cpu())
                    all_masks.append(mask_v.squeeze().cpu())
                    # Log validation visuals for the first batch only
                    if len(ious) == 1:
                        def _denorm(t):
                            # t: (C,H,W) normalized with ImageNet mean/std
                            mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
                            img = (t * std) + mean
                            return img.clamp(0.0, 1.0)

                        pre_img = _denorm(pre_v[0])
                        post_img = _denorm(post_v[0])
                        pred_img = probs_v[0].repeat(3, 1, 1)
                        mask_img = mask_v[0].repeat(3, 1, 1)

                        grid = vutils.make_grid([pre_img, post_img, pred_img, mask_img], nrow=4)
                        writer.add_image("val/sample_grid", grid, epoch)
                        # Save grid to outputs/val_images
                        outputs_dir = Path("outputs")
                        outputs_dir.mkdir(parents=True, exist_ok=True)
                        grid_np = (grid.cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                        Image.fromarray(grid_np).save(outputs_dir / f"val_sample_epoch_{epoch:03d}.png")

            val_iou = float(sum(ious) / len(ious)) if ious else 0.0
            val_dice = float(sum(dices) / len(dices)) if dices else 0.0
            val_acc = float(sum(accs) / len(accs)) if accs else 0.0

            writer.add_scalar("val/loss", val_loss / len(val_loader), epoch)
            writer.add_scalar("val/iou", val_iou, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            model.train()

            # Threshold optimization
            best_threshold = None
            if train_cfg.get("optimize_threshold", False) and (all_probs and all_masks):
                thresholds = train_cfg.get("threshold_grid", [0.3, 0.4, 0.5, 0.6])
                best_iou = -1.0
                for t in thresholds:
                    ious_t = []
                    for p, m in zip(all_probs, all_masks):
                        pred_bin = (p > t).float()
                        inter = (pred_bin * m).sum().item()
                        union = pred_bin.sum().item() + m.sum().item() - inter
                        iou = inter / (union + 1e-6)
                        ious_t.append(iou)
                    mean_iou = float(sum(ious_t) / len(ious_t))
                    if mean_iou > best_iou:
                        best_iou = mean_iou
                        best_threshold = t
                writer.add_scalar("val/best_threshold", best_threshold, epoch)
                writer.add_scalar("val/best_threshold_iou", best_iou, epoch)
                # Save threshold into metadata for best checkpoint
                current_metadata = {"best_threshold": best_threshold}
            else:
                current_metadata = None

        writer.add_scalar("train/loss", epoch_loss / len(train_loader), epoch)
        writer.add_scalar("train/pos_weight", pos_w, epoch)
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        ckpt_path = Path("checkpoints") / f"epoch_{epoch:03d}.pth"
        save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, metadata={**saved_metadata, **({"epoch": epoch} if saved_metadata else {})})
        shutil.copy2(ckpt_path, Path("checkpoints") / "latest.pth")

        if val_iou is not None and (best_val_iou is None or val_iou > best_val_iou):
            best_val_iou = val_iou
            # merge any threshold metadata and saved_metadata
            final_metadata = {**saved_metadata, **(current_metadata or {})}
            final_metadata["epoch"] = epoch
            save_checkpoint(Path("checkpoints") / "best.pth", epoch, model, optimizer, scaler, metadata=final_metadata)

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        report = {"epoch": epoch, "train_loss": epoch_loss / len(train_loader), "val_iou": val_iou, "val_dice": val_dice, "val_accuracy": val_acc}
        if current_metadata and "best_threshold" in current_metadata:
            report["best_threshold"] = current_metadata["best_threshold"]
        with open(outputs_dir / "performance_report.json", "w") as f:
            json.dump(report, f, indent=2)

        t1 = time.time()
        print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader):.4f} (t={t1-t0:.1f}s)")
        if val_iou is not None:
            print(f"  Val IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}")
        else:
            print("  No validation dataset provided.")

    writer.close()


if __name__ == "__main__":
    train()
