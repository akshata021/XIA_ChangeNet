"""Training script for XAl-ChangeNet."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader

from models.siamese_unet import SiameseResNet18UNet
from utils.data_loader import XBDChangeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XAl-ChangeNet")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--val_pairs", type=Path, default=None, help="Optional validation pairs manifest (JSON)")
    parser.add_argument("--logdir", type=Path, default=Path("runs"), help="TensorBoard runs dir")
    return parser.parse_args()

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_pairs_file(root: Path) -> Path:
    matches = sorted(root.glob("pairs_*.json"))
    if not matches:
        raise FileNotFoundError(f"No pairs manifest found in {root}")
    return matches[0]


def build_dataloader(cfg: Dict[str, Any], pairs_file: Path | None = None, augment: bool = True, shuffle: bool = True) -> DataLoader:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    pairs_path = Path(pairs_file) if pairs_file else Path(data_cfg.get("pairs_file") or get_pairs_file(root))
    dataset = XBDChangeDataset(root=root, pairs_file=pairs_path, img_size=data_cfg.get("img_size", 512), augment=augment)
    batch_size = data_cfg.get("batch_size", 2) if augment else data_cfg.get("val_batch_size", 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=augment,
    )


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num + eps) / den


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer: optim.Optimizer, scaler: amp.GradScaler):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(payload, path)


def train():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseResNet18UNet()
    model = model.to(device)

    dataloader = build_dataloader(cfg)

    train_cfg = cfg["training"]
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-4))
    scaler = amp.GradScaler()

    bce = nn.BCEWithLogitsLoss()
    accum_steps = train_cfg.get("accum_steps", 1)
    # validation dataloader: use pairs from CLI if provided, otherwise fallback to config key 'val_pairs' if present
    val_pairs_cli = args.val_pairs
    val_pairs_cfg = cfg["data"].get("val_pairs")
    val_loader = None
    if val_pairs_cli or val_pairs_cfg:
        val_pairs_path = Path(val_pairs_cli) if val_pairs_cli else Path(val_pairs_cfg)
        val_batch = cfg["data"].get("val_batch_size", max(1, cfg["data"].get("batch_size", 2)))
        val_ds = XBDChangeDataset(root=Path(cfg["data"]["root"]), pairs_file=val_pairs_path, img_size=cfg["data"].get("img_size", 512), augment=False)
        val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=cfg["data"].get("num_workers", 2), pin_memory=True)

    global_step = 0
    for epoch in range(1, train_cfg.get("epochs", 20) + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(dataloader, start=1):
            pre, post, mask = [x.to(device) for x in batch]
            with amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(pre, post)
                loss = bce(logits, mask) + dice_loss(logits, mask).mean()
                loss = loss / accum_steps
            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            global_step += 1

        ckpt_path = Path("checkpoints") / f"epoch_{epoch:03d}.pth"
        save_checkpoint(ckpt_path, epoch, model, optimizer, scaler)
        shutil.copy2(ckpt_path, Path("checkpoints") / "latest.pth")
        print(f"Epoch {epoch} loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    train()
